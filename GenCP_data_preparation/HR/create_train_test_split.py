"""Create train/test dataset splits for GenCP HR training.

Scans all product subdirectories under --input_dir and collects images
from the specified subfolder paths, then splits into train/test.

Supports two modes:

1. Standard (aligned) mode:
   Collects AB merged image pairs from a subfolder (default: AB_TCI_5m)
   in each product directory and splits into train/ and test/.

2. Topography mode:
   Collects images from three subfolder paths (A, B, TOPO) in each
   product directory and splits into train/{A,B,TOPO}/ and test/{A,B,TOPO}/.

Usage examples:

  # Standard mode - collect AB pairs from all products
  python -m GenCP_data_preparation.HR.create_train_test_split \
      --input_dir /path/to/pipeline_output \
      --output_dir /path/to/dataset \
      --subfolder_ab AB_TCI_5m

  # Topography mode - collect A, B, TOPO from all products
  python -m GenCP_data_preparation.HR.create_train_test_split \
      --input_dir /path/to/pipeline_output \
      --output_dir /path/to/dataset \
      --mode topography \
      --subfolder_a OSM_5m/image_filled/10m \
      --subfolder_b TCI/S2_crop_extended \
      --subfolder_topo TOPO

  # Use symlinks instead of copying (saves disk space)
  python -m GenCP_data_preparation.HR.create_train_test_split \
      --input_dir /path/to/pipeline_output \
      --output_dir /path/to/dataset \
      --symlink
"""

import argparse
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

IMG_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff',
}


def list_images(directory: str) -> List[str]:
    """Return sorted list of image filenames in a directory."""
    return sorted(
        f for f in os.listdir(directory)
        if Path(f).suffix.lower() in IMG_EXTENSIONS
    )


def collect_from_products(
    input_dir: str,
    subfolder: str,
) -> List[Tuple[str, str]]:
    """Collect (full_path, filename) pairs from subfolder in each product dir."""
    results = []
    for product in sorted(os.listdir(input_dir)):
        src_dir = os.path.join(input_dir, product, subfolder)
        if not os.path.isdir(src_dir):
            continue
        for fname in list_images(src_dir):
            results.append((os.path.join(src_dir, fname), fname))
    return results


def place_file(src: str, dst: str, symlink: bool) -> None:
    """Copy or symlink a file from src to dst."""
    if symlink:
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def split_standard(
    input_dir: str,
    output_dir: str,
    subfolder_ab: str,
    test_ratio: float,
    seed: int,
    symlink: bool,
) -> None:
    """Collect AB pairs from all products and split into train/ and test/."""
    files = collect_from_products(input_dir, subfolder_ab)
    if not files:
        raise FileNotFoundError(
            f"No images found in any product's {subfolder_ab}/ under {input_dir}"
        )

    random.seed(seed)
    random.shuffle(files)
    n_test = max(1, int(len(files) * test_ratio))

    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for src, fname in files[n_test:]:
        place_file(src, os.path.join(train_dir, fname), symlink)
    for src, fname in files[:n_test]:
        place_file(src, os.path.join(test_dir, fname), symlink)

    logger.info(
        "Standard split: %d train, %d test in %s",
        len(files) - n_test, n_test, output_dir,
    )


def split_topography(
    input_dir: str,
    output_dir: str,
    subfolder_a: str,
    subfolder_b: str,
    subfolder_topo: str,
    test_ratio: float,
    seed: int,
    symlink: bool,
) -> None:
    """Collect A, B, TOPO from all products and split into train/test.

    Files are matched by stem across the three subfolder types within each
    product directory. Only files present in all three are included.
    """
    # Collect per-product: {stem: full_path} for each role
    all_a: Dict[str, str] = {}
    all_b: Dict[str, str] = {}
    all_topo: Dict[str, str] = {}

    for product in sorted(os.listdir(input_dir)):
        product_dir = os.path.join(input_dir, product)
        if not os.path.isdir(product_dir):
            continue

        dir_a = os.path.join(product_dir, subfolder_a)
        dir_b = os.path.join(product_dir, subfolder_b)
        dir_topo = os.path.join(product_dir, subfolder_topo)

        if not (os.path.isdir(dir_a) and os.path.isdir(dir_b) and os.path.isdir(dir_topo)):
            logger.warning("Skipping product %s: missing one or more subfolders", product)
            continue

        for f in list_images(dir_a):
            all_a[Path(f).stem] = os.path.join(dir_a, f)
        for f in list_images(dir_b):
            all_b[Path(f).stem] = os.path.join(dir_b, f)
        for f in list_images(dir_topo):
            all_topo[Path(f).stem] = os.path.join(dir_topo, f)

    common_stems = sorted(all_a.keys() & all_b.keys() & all_topo.keys())
    if not common_stems:
        raise FileNotFoundError(
            f"No matching files across A/B/TOPO in any product under {input_dir}. "
            f"Found {len(all_a)} A, {len(all_b)} B, {len(all_topo)} TOPO total."
        )

    skipped = (set(all_a) | set(all_b) | set(all_topo)) - set(common_stems)
    if skipped:
        logger.warning(
            "Skipping %d files not present in all three folders (showing first 5): %s",
            len(skipped), ", ".join(sorted(skipped)[:5]),
        )

    random.seed(seed)
    random.shuffle(common_stems)
    n_test = max(1, int(len(common_stems) * test_ratio))
    test_stems = common_stems[:n_test]
    train_stems = common_stems[n_test:]

    # Create output directories
    for phase in ("train", "test"):
        for subfolder in ("A", "B", "TOPO"):
            os.makedirs(os.path.join(output_dir, phase, subfolder), exist_ok=True)

    src_maps = {"A": all_a, "B": all_b, "TOPO": all_topo}

    for stem in train_stems:
        for folder, mapping in src_maps.items():
            src = mapping[stem]
            place_file(src, os.path.join(output_dir, "train", folder, Path(src).name), symlink)

    for stem in test_stems:
        for folder, mapping in src_maps.items():
            src = mapping[stem]
            place_file(src, os.path.join(output_dir, "test", folder, Path(src).name), symlink)

    logger.info(
        "Topography split: %d train, %d test in %s",
        len(train_stems), len(test_stems), output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create train/test dataset splits for GenCP HR training.",
    )
    parser.add_argument(
        "--input_dir", "-i", required=True,
        help="Root directory containing all S2 product subdirectories",
    )
    parser.add_argument(
        "--output_dir", "-o", required=True,
        help="Output directory where train/ and test/ will be created",
    )
    parser.add_argument(
        "--mode", "-m", choices=["standard", "topography"], default="standard",
        help="Dataset mode (default: standard)",
    )
    parser.add_argument(
        "--test_ratio", "-r", type=float, default=0.2,
        help="Fraction of data to use for testing (default: 0.2)",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--symlink", action="store_true",
        help="Create symbolic links instead of copying files (saves disk space)",
    )

    # Subfolder names (relative paths within each product directory)
    parser.add_argument(
        "--subfolder_ab", default="AB_TCI_5m",
        help="[standard] Subfolder containing AB merged pairs (default: AB_TCI_5m)",
    )
    parser.add_argument(
        "--subfolder_a", default="OSM_5m/image_filled/10m",
        help="[topography] Subfolder for OSM rasters (default: OSM_5m/image_filled/10m)",
    )
    parser.add_argument(
        "--subfolder_b", default="TCI/S2_crop_extended",
        help="[topography] Subfolder for reference S2 crops (default: TCI/S2_crop_extended)",
    )
    parser.add_argument(
        "--subfolder_topo", default="TOPO",
        help="[topography] Subfolder for topography crops (default: TOPO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.mode == "standard":
        split_standard(
            args.input_dir, args.output_dir, args.subfolder_ab,
            args.test_ratio, args.seed, args.symlink,
        )
    else:
        split_topography(
            args.input_dir, args.output_dir,
            args.subfolder_a, args.subfolder_b, args.subfolder_topo,
            args.test_ratio, args.seed, args.symlink,
        )


if __name__ == "__main__":
    main()
