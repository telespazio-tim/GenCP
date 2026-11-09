"""Create train/test dataset splits for GenCP HR training — SEASONAL version.

Produces a dataset in pix2pix paired_folders format with seasons:
    output_dir/
        train/
            A_summer/   (OSM rasters)
            B_summer/   (real S2)
            A_winter/
            B_winter/
        test/
            A_summer/, B_summer/, A_winter/, B_winter/

Inputs are two separate directories (summer and winter), each holding the
product folders written by ``dataset_creation`` (OSM rasters in
``OSM_5m/image_filled/10m/``, S2 crops in ``TCI/S2_crop_extended/``).

The split is computed once over the set of geographic keys (T<MGRS>_tile_NNNN_MMMM
or any common stem) and applied identically to summer and winter, so that
the same tile ends up in the same phase (train OR test) for both seasons.

Usage:

  # Standard usage (paired_folders format)
  python -m GenCP_data_preparation.HR.create_seasonal_train_test_split \
      --summer_input /path/to/summer_products \
      --winter_input /path/to/winter_products \
      --output_dir /path/to/seasonal_dataset \
      --test_ratio 0.2 \
      --subfolder_a OSM_5m/image_filled/10m \
      --subfolder_b TCI/S2_crop_extended

  # Save disk space with symlinks
  python -m GenCP_data_preparation.HR.create_seasonal_train_test_split \
      --summer_input /path/to/summer_products \
      --winter_input /path/to/winter_products \
      --output_dir /path/to/seasonal_dataset \
      --symlink

The same image stem (e.g. "T31TCJ_tile_0001_0001.png") is expected in both summer
and winter inputs. A trailing "_summer" / "_winter" in the stem is ignored for
matching, so "T30TXT_tile_0000_0000_summer.png" pairs with
"T30TXT_tile_0000_0000_winter.png". Tiles present in only one season are
excluded (with a warning).
"""

import argparse
import logging
import os
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

IMG_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff',
}

# Season suffix removed from stems before matching summer and winter tiles
SEASON_SUFFIX_RE = re.compile(r"_(summer|winter)$", re.IGNORECASE)


def tile_key(stem: str) -> str:
    """Geographic key of a tile: its stem without a trailing season suffix."""
    return SEASON_SUFFIX_RE.sub("", stem)


def list_images(directory: str) -> List[str]:
    """Return sorted list of image filenames in a directory."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        f for f in os.listdir(directory)
        if Path(f).suffix.lower() in IMG_EXTENSIONS
    )


def collect_role(
    root_dir: str,
    subfolder: str,
) -> Dict[str, str]:
    """Scan all product subdirectories under root_dir for a given role subfolder.

    Returns:
        Dict[tile key -> full_path], the key being the stem without a season
        suffix (see :func:`tile_key`).
    """
    result: Dict[str, str] = {}
    if not os.path.isdir(root_dir):
        return result

    for product in sorted(os.listdir(root_dir)):
        product_dir = os.path.join(root_dir, product)
        if not os.path.isdir(product_dir):
            continue
        src_dir = os.path.join(product_dir, subfolder)
        if not os.path.isdir(src_dir):
            continue
        for fname in list_images(src_dir):
            stem = tile_key(Path(fname).stem)
            full_path = os.path.join(src_dir, fname)
            if stem in result:
                logger.warning(
                    "Duplicate stem '%s' across products: keeping %s, ignoring %s",
                    stem, result[stem], full_path,
                )
                continue
            result[stem] = full_path
    return result


def place_file(src: str, dst: str, symlink: bool) -> None:
    """Copy or symlink a file from src to dst."""
    if symlink:
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def split_seasonal(
    summer_input: str,
    winter_input: str,
    output_dir: str,
    subfolder_a: str,
    subfolder_b: str,
    test_ratio: float,
    seed: int,
    symlink: bool,
) -> None:
    """Build the seasonal paired_folders dataset with train/test split."""
    # Collect 4 indices: A/B x summer/winter
    logger.info("Scanning summer A from %s/%s", summer_input, subfolder_a)
    a_summer = collect_role(summer_input, subfolder_a)
    logger.info("Scanning summer B from %s/%s", summer_input, subfolder_b)
    b_summer = collect_role(summer_input, subfolder_b)
    logger.info("Scanning winter A from %s/%s", winter_input, subfolder_a)
    a_winter = collect_role(winter_input, subfolder_a)
    logger.info("Scanning winter B from %s/%s", winter_input, subfolder_b)
    b_winter = collect_role(winter_input, subfolder_b)

    logger.info(
        "Found: A_summer=%d, B_summer=%d, A_winter=%d, B_winter=%d",
        len(a_summer), len(b_summer), len(a_winter), len(b_winter),
    )

    # Compute intersection of all 4 sets (tiles present in all 4 roles)
    common_stems: Set[str] = (
        set(a_summer) & set(b_summer) & set(a_winter) & set(b_winter)
    )
    if not common_stems:
        raise FileNotFoundError(
            "No common tile stems across A_summer/B_summer/A_winter/B_winter. "
            f"Got {len(a_summer)} / {len(b_summer)} / {len(a_winter)} / "
            f"{len(b_winter)} respectively. Check --subfolder_a/--subfolder_b "
            f"('{subfolder_a}' / '{subfolder_b}') and that both seasons cover "
            "the same tiles."
        )

    common_sorted = sorted(common_stems)

    # Log skipped tiles (present in some but not all 4)
    all_seen = set(a_summer) | set(b_summer) | set(a_winter) | set(b_winter)
    skipped = all_seen - common_stems
    if skipped:
        logger.warning(
            "Skipping %d tiles not present in all 4 (A/B)x(summer/winter) sets. "
            "First 5: %s",
            len(skipped), ", ".join(sorted(skipped)[:5]),
        )

    # Split applied to common stems (same split for both seasons)
    random.seed(seed)
    shuffled = common_sorted.copy()
    random.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_ratio))
    test_stems = set(shuffled[:n_test])
    train_stems = set(shuffled[n_test:])

    logger.info(
        "Split: %d train tiles, %d test tiles (same split applied to both seasons)",
        len(train_stems), len(test_stems),
    )

    # Create output structure
    for phase in ("train", "test"):
        for folder in ("A_summer", "B_summer", "A_winter", "B_winter"):
            os.makedirs(os.path.join(output_dir, phase, folder), exist_ok=True)

    # Place files
    role_mapping = {
        "A_summer": a_summer,
        "B_summer": b_summer,
        "A_winter": a_winter,
        "B_winter": b_winter,
    }

    n_placed_train = 0
    n_placed_test = 0
    for stem in common_sorted:
        phase = "test" if stem in test_stems else "train"
        for folder_name, idx in role_mapping.items():
            src = idx[stem]
            dst = os.path.join(
                output_dir, phase, folder_name, Path(src).name,
            )
            place_file(src, dst, symlink)
        if phase == "train":
            n_placed_train += 1
        else:
            n_placed_test += 1

    logger.info(
        "Done: placed %d train tiles and %d test tiles (4 files each) in %s",
        n_placed_train, n_placed_test, output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create train/test seasonal split (pix2pix paired_folders).",
    )
    parser.add_argument(
        "--summer_input", required=True,
        help="Root directory containing all summer S2 product subdirectories",
    )
    parser.add_argument(
        "--winter_input", required=True,
        help="Root directory containing all winter S2 product subdirectories",
    )
    parser.add_argument(
        "--output_dir", "-o", required=True,
        help="Output directory where train/{A,B}_{summer,winter}/ will be created",
    )
    parser.add_argument(
        "--test_ratio", "-r", type=float, default=0.2,
        help="Fraction of tiles to use for testing (default: 0.2)",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--symlink", action="store_true",
        help="Create symbolic links instead of copying files (saves disk space)",
    )
    parser.add_argument(
        "--subfolder_a", default="OSM_5m/image_filled/10m",
        help="Path of the OSM rasters inside each product "
             "(default: OSM_5m/image_filled/10m; OSM_5m/image without --clc)",
    )
    parser.add_argument(
        "--subfolder_b", default="TCI/S2_crop_extended",
        help="Path of the S2 crops inside each product "
             "(default: TCI/S2_crop_extended; <band>/S2_crop with --no_buffer)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    split_seasonal(
        summer_input=args.summer_input,
        winter_input=args.winter_input,
        output_dir=args.output_dir,
        subfolder_a=args.subfolder_a,
        subfolder_b=args.subfolder_b,
        test_ratio=args.test_ratio,
        seed=args.seed,
        symlink=args.symlink,
    )


if __name__ == "__main__":
    main()