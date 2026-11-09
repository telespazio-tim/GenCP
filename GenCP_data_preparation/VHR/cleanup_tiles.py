"""

cleanup_tiles.py – Black Tile Cleanup Tool

This script checks raster tiles and removes tile pairs whose black pixel ratio
is greater than or equal to a user-defined threshold.

The script is designed for paired raster and OSM mask tile datasets. For each
raster tile, the percentage of black pixels is calculated. If this percentage
exceeds the threshold, the raster tile and its corresponding OSM mask tile are
either moved to a "deleted" folder or permanently deleted.

Input parameters:
--base_dir:
Main directory containing tile folders.
Each tile folder is expected to contain "raster" and "osm" subfolders.

--threshold:
    Black pixel ratio threshold between 0 and 1.
    Default value: 0.01, corresponding to 1%.

--delete:
    If specified, matching tile pairs are permanently deleted.
    If not specified, they are moved to a "deleted" folder.

--folders:
    Optional list of specific subfolder names to process.
    If not provided, all subfolders under base_dir are processed.

Expected input folder structure:
base_dir/
└── raster_name/
├── raster/
│ ├── raster_name_patch_0.tif
│ ├── raster_name_patch_1.tif
│ └── ...
└── osm/
├── raster_name_mask_0.tif
├── raster_name_mask_1.tif
└── ...

Outputs:
If --delete is not used:
Removed raster and OSM mask tiles are moved to:

    raster_name/
    └── deleted/
        ├── raster/
        └── osm/

If --delete is used:
    Removed raster and OSM mask tiles are permanently deleted.

    
Usage examples:

  # Basic usage:
  python cleanup_tiles.py --base_dir "C:\...\tiles"

  # Change the treshold:
  python cleanup_tiles.py --base_dir "C:\...\tiles" --threshold 0.05

  # Permanenetly delete tiles instead of moving them:
  python cleanup_tiles.py --base_dir "C:\...\tiles" --delete

  # Process only specific subfolders:
  python cleanup_tiles.py --base_dir "C:\...\tiles" --folders raster1 raster2
"""

import os
import sys
import shutil
import argparse
from osgeo import gdal
import numpy as np

# Make the shared GenCP_data_preparation package importable when this file is
# run as a standalone script (repo root is three levels up from this module:
# genCP/GenCP_data_preparation/VHR/<this file>).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from GenCP_data_preparation.common.tile_filters import black_pixel_ratio

# ════════════════════════════════════════════════════════════
#  ARGPARSE
# ════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(
    description="Moves or deletes tiles whose black pixel ratio exceeds the specified threshold."
)

parser.add_argument(
    "--base_dir", required=True,
    help="Main directory containing the tile folders."
)
parser.add_argument(
    "--threshold", type=float, default=0.01,
    help="Black pixel ratio threshold between 0 and 1 (default: 0.01 → 1%)"
)
parser.add_argument(
    "--delete", action="store_true",
    help="If specified, tiles are permanently deleted instead of being moved to the 'deleted' folder."
)
parser.add_argument(
    "--folders", nargs="*", default=None,
    help="Names of specific subfolders under base_dir to process. "
        "If not provided, all subfolders under base_dir will be processed."
)

args = parser.parse_args()

base_dir        = args.base_dir
black_threshold = args.threshold
direct_delete   = args.delete

# ════════════════════════════════════════════════════════════
#  Processing
# ════════════════════════════════════════════════════════════

print(f"Main directory  : {base_dir}")
print(f"Treshold       : {black_threshold:.2%}  (tiles with this ratio or more black pixels will be processed)")
print(f"Mod        : {'Permanent DELETE' if direct_delete else 'MOVE to deleted folder'}")
print("-" * 55)

if args.folders:
    sub_folders = [f for f in args.folders if os.path.isdir(os.path.join(base_dir, f))]
    missing = set(args.folders) - set(sub_folders)
    if missing:
        print(f"[WARNING] folders not found: {missing}")
else:
    sub_folders = [
        f for f in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, f))
    ]

total_removed = 0

for folder in sub_folders:
    current_path = os.path.join(base_dir, folder)
    raster_dir   = os.path.join(current_path, "raster")
    osm_dir      = os.path.join(current_path, "osm")
    deleted_dir  = os.path.join(current_path, "deleted")

    print(f"\nChecking folder: {folder}")

    if not os.path.exists(raster_dir):
        print("  [SKIP] raster/ subfolder does not exist.")
        continue

    if not direct_delete:
        os.makedirs(os.path.join(deleted_dir, "raster"), exist_ok=True)
        os.makedirs(os.path.join(deleted_dir, "osm"),    exist_ok=True)

    files = [f for f in os.listdir(raster_dir) if f.endswith(".tif")]
    count_deleted = 0

    for f in files:
        raster_path = os.path.join(raster_dir, f)

        ds = gdal.Open(raster_path)
        if ds is None:
            continue

        data = ds.ReadAsArray()
        black_ratio = black_pixel_ratio(data)
        ds = None

        if black_ratio < black_threshold:
            continue

        # Process the raster tile
        if direct_delete:
            os.remove(raster_path)
        else:
            shutil.move(raster_path, os.path.join(deleted_dir, "raster", f))

        # Process the matching OSM mask tile
        osm_file_name = f.replace("_patch_", "_mask_")
        osm_path      = os.path.join(osm_dir, osm_file_name)

        if os.path.exists(osm_path):
            if direct_delete:
                os.remove(osm_path)
            else:
                shutil.move(osm_path, os.path.join(deleted_dir, "osm", osm_file_name))

        count_deleted += 1

    action = "deleted" if direct_delete else "move to the 'deleted' folder"
    print(f"  Completed: {count_deleted} tiles {action}  (total cehecked: {len(files)} tiles)")
    total_removed += count_deleted

print("-" * 55)
print(f"All processing completed  |  Total removed tiles: {total_removed}")