"""
single_class_tiles.py
Filter (OSM, S2) pairs where the OSM raster contains a single semantic class.
Such tiles do not bring useful information for ControlNet training.

Single-class detection: we look at the dominant pixel ratio in the OSM raster.
If > threshold (default 99%), the tile is considered single-class and removed.

Uses the OSM color palette of the HR pipeline (``config.color_dict`` plus the
building colour). Each pixel is assigned to the closest palette colour if it is
within ``--color_tolerance`` (lossy-compressed rasters are slightly off-palette);
pixels further away are ignored in the ratio computation.

Works on standard splits (A/, B/, optionally TOPO/) and seasonal splits
(A_summer/, B_summer/, A_winter/, B_winter/). When a tile is removed, its file is
deleted from every one of these folders, for both seasons.

Usage:
    python -m GenCP_data_preparation.HR.single_class_tiles \
        --split_dir /path/to/split_output \
        --threshold 0.99 \
        --dry_run

    # Once validated in dry_run, re-run without the flag to actually delete
    python -m GenCP_data_preparation.HR.single_class_tiles \
        --split_dir /path/to/split_output \
        --threshold 0.99
"""
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import rasterio
from PIL import Image, ImageColor

from .config import color_dict
from .create_seasonal_train_test_split import tile_key


# OSM palette = colours the HR rasterizer can write: config.color_dict, the
# building colour (osm_extractor.process_tags_data) and "yellow_farm", kept from
# the legacy palette (genCP_HR_osm_colors.py) for older datasets.
COLOR_DICT_HEX = {
    **color_dict,
    "building": "brown",
    "yellow_farm": "#ecffb3",
}

DEFAULT_COLOR_TOLERANCE = 8
"""Max per-channel difference (0-255) for a pixel to count as a palette colour."""

# Folders handled in a split directory; OSM (A) folders are the ones analysed.
ROLE_FOLDERS = ("A", "B", "TOPO", "A_summer", "B_summer", "A_winter", "B_winter")
OSM_FOLDERS = ("A", "A_summer", "A_winter")


def hex_to_rgb(color):
    """RGB tuple of a hex string or a colour name ("white", "brown", ...)."""
    return ImageColor.getrgb(color)[:3]


def build_palette():
    """Return the palette as a (K, 3) int array of unique RGB colours."""
    return np.array(sorted({hex_to_rgb(c) for c in COLOR_DICT_HEX.values()}), dtype=np.int64)


def read_raster_rgb(path):
    """Read a raster (TIF/PNG) as an (H, W, 3) uint8 array."""
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        with rasterio.open(path) as src:
            arr = src.read()
            if arr.shape[0] >= 3:
                arr = arr[:3]
            else:
                arr = np.repeat(arr, 3, axis=0)
            arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr
    return np.array(Image.open(path).convert("RGB"))


def dominant_class_ratio(osm_array, palette, tolerance=DEFAULT_COLOR_TOLERANCE):
    """
    Return (dominant_ratio, dominant_rgb, n_classes) for an OSM raster.

    Each pixel is assigned to the closest palette colour (max per-channel
    difference) if that difference is <= ``tolerance``; other pixels are ignored.

    dominant_ratio: fraction of pixels (among those in the palette) belonging
                    to the most frequent color.
    dominant_rgb:   (r, g, b) of the dominant palette color.
    n_classes:      number of palette classes present in the tile.

    If less than 50% of pixels are in the palette, returns (None, None, 0).
    """
    flat = osm_array.reshape(-1, 3).astype(np.int64)
    unique_colors, counts_per_color = np.unique(flat, axis=0, return_counts=True)

    # Closest palette colour for each distinct pixel colour
    dist = np.abs(unique_colors[:, None, :] - palette[None, :, :]).max(axis=2)
    nearest = dist.argmin(axis=1)
    in_palette = dist[np.arange(len(unique_colors)), nearest] <= tolerance

    class_counts = np.bincount(nearest[in_palette], weights=counts_per_color[in_palette],
                               minlength=len(palette))
    total = class_counts.sum()

    if total < 0.5 * len(flat):
        # Too few palette pixels: suspect tile, ignore (None)
        return None, None, 0

    dominant = int(class_counts.argmax())
    ratio = class_counts[dominant] / total
    return ratio, tuple(int(v) for v in palette[dominant]), int((class_counts > 0).sum())


def find_image_files(directory: Path):
    """List TIF/PNG images in a directory."""
    return sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in {".tif", ".tiff", ".png"}
    )


def collect_tiles(split_dir: Path):
    """
    Group the files of a split directory by tile.

    Files are matched by tile key (stem without season suffix) across every
    role folder present (A, B, TOPO and their seasonal variants).
    Returns dict {key: (osm_path, [all files of the tile])}, for tiles that
    have an OSM file and at least one other file.
    """
    roles = [r for r in ROLE_FOLDERS if (split_dir / r).is_dir()]
    index = {r: {tile_key(f.stem): f for f in find_image_files(split_dir / r)} for r in roles}
    osm_roles = [r for r in roles if r in OSM_FOLDERS]

    tiles = {}
    for key in sorted(set().union(*(index[r] for r in osm_roles))):
        files = [index[r][key] for r in roles if key in index[r]]
        missing = [r for r in roles if key not in index[r]]
        osm_path = next(index[r][key] for r in osm_roles if key in index[r])
        if missing:
            print(f"  WARNING: {osm_path.name} has no match in {', '.join(missing)}")
        if len(files) > 1:
            tiles[key] = (osm_path, files)
    return tiles


def process_split(split_dir: Path, threshold: float, dry_run: bool, palette,
                  tolerance=DEFAULT_COLOR_TOLERANCE):
    """Process a sub-directory (train or test)."""
    roles = [r for r in ROLE_FOLDERS if (split_dir / r).is_dir()]
    if not any(r in OSM_FOLDERS for r in roles) or all(r in OSM_FOLDERS for r in roles):
        print(f"  Skipping {split_dir}: no A/ (or A_<season>/) and B/ folders")
        return

    print(f"\n=== Processing {split_dir} ({', '.join(roles)}) ===")
    pairs = collect_tiles(split_dir)
    print(f"  {len(pairs)} tiles found")

    n_kept = 0
    n_removed_mono = 0
    n_removed_suspect = 0
    removed_files = []
    dominant_distribution = Counter()

    for i, (stem, (a_path, tile_files)) in enumerate(pairs.items(), 1):
        if i % 200 == 0:
            print(f"  {i}/{len(pairs)}...")

        try:
            osm = read_raster_rgb(a_path)
        except Exception as e:
            print(f"  ERROR reading {a_path.name}: {e}")
            continue

        ratio, dominant_rgb, n_classes = dominant_class_ratio(osm, palette, tolerance)

        if ratio is None:
            # Suspect tile (too many out-of-palette pixels)
            n_removed_suspect += 1
            removed_files.append((a_path, tile_files, "suspect_out_of_palette"))
            continue

        if ratio >= threshold:
            # Single-class, to be removed
            n_removed_mono += 1
            dominant_distribution[dominant_rgb] += 1
            removed_files.append((a_path, tile_files, f"mono_{dominant_rgb}_{ratio:.3f}"))
        else:
            n_kept += 1

    print(f"\n  Results for {split_dir.name}:")
    print(f"    Kept:                          {n_kept}")
    print(f"    Removed (single-class):        {n_removed_mono}")
    print(f"    Removed (suspect):             {n_removed_suspect}")

    if dominant_distribution:
        print("\n  Dominant classes of removed single-class tiles:")
        for rgb, count in dominant_distribution.most_common():
            # Find the color name
            color_name = "?"
            for name, hex_str in COLOR_DICT_HEX.items():
                if hex_to_rgb(hex_str) == rgb:
                    color_name = name
                    break
            print(f"    {rgb} ({color_name}): {count}")

    if dry_run:
        print(f"\n  [DRY RUN] No files deleted. {len(removed_files)} would be deleted.")
        if removed_files:
            print("  Examples (first 5):")
            for a, _, reason in removed_files[:5]:
                print(f"    - {a.name} ({reason})")
    else:
        print(f"\n  Deleting {len(removed_files)} tiles...")
        for a, tile_files, reason in removed_files:
            for f in tile_files:
                try:
                    # Handle symlinks the same way as regular files
                    if f.is_symlink() or f.exists():
                        f.unlink()
                except Exception as e:
                    print(f"    ERROR deleting {f}: {e}")
        print("  Deletion completed.")

    return n_kept, n_removed_mono, n_removed_suspect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", type=Path, required=True,
                        help="Directory containing train/ and test/, each with A/ and B/ "
                             "(optionally TOPO/) or A_summer/, B_summer/, A_winter/, B_winter/")
    parser.add_argument("--threshold", type=float, default=0.99,
                        help="Dominant pixel ratio above which a tile is "
                             "considered single-class (default: 0.99)")
    parser.add_argument("--color_tolerance", type=int, default=DEFAULT_COLOR_TOLERANCE,
                        help="Max per-channel difference for a pixel to match a palette "
                             f"colour (default: {DEFAULT_COLOR_TOLERANCE}; 0 = exact match)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Do not delete anything, just show what would be deleted")
    parser.add_argument("--only", choices=["train", "test"], default=None,
                        help="Process only one split (train or test)")
    args = parser.parse_args()

    palette = build_palette()
    print(f"Palette: {len(palette)} unique colors, tolerance {args.color_tolerance}")
    print(f"Single-class threshold: dominant ratio >= {args.threshold}")
    if args.dry_run:
        print("DRY RUN mode (no actual deletion)")

    splits = [args.only] if args.only else ["train", "test"]

    totals = {"kept": 0, "mono": 0, "suspect": 0}
    for split_name in splits:
        split_dir = args.split_dir / split_name
        if not split_dir.exists():
            print(f"\nSkipping {split_dir}: directory does not exist")
            continue
        result = process_split(
            split_dir,
            threshold=args.threshold,
            dry_run=args.dry_run,
            palette=palette,
            tolerance=args.color_tolerance,
        )
        if result:
            kept, mono, suspect = result
            totals["kept"] += kept
            totals["mono"] += mono
            totals["suspect"] += suspect

    print(f"\n{'=' * 50}")
    print(f"TOTAL across {', '.join(splits)}:")
    print(f"  Kept:       {totals['kept']}")
    print(f"  Mono:       {totals['mono']}")
    print(f"  Suspect:    {totals['suspect']}")
    if totals["kept"] + totals["mono"] + totals["suspect"] > 0:
        pct_removed = (totals["mono"] + totals["suspect"]) / (
            totals["kept"] + totals["mono"] + totals["suspect"]
        ) * 100
        print(f"  % removed:  {pct_removed:.1f}%")


if __name__ == "__main__":
    main()