"""

merge_shp.py – OSM Shapefile Merger

This script scans the input directory and its subfolders, finds OSM shapefiles
with predefined layer names, and merges shapefiles belonging to the same layer
into a single output shapefile.

Duplicate features are removed using both osm_id and geometry when osm_id is
available. If osm_id is not available, geometry is used for duplicate removal.
This prevents data loss when the same osm_id has multiple geometry parts.

Input parameters:
    --base_dir:
        Root directory to be scanned.

    --output_dir:
        Folder where the merged shapefiles will be saved.

    --categories:
        Optional short names of layers to be merged.
        If not provided, all predefined layers are processed.

    --exclude:
        Optional short names of layers to be excluded.

Outputs:
    - One merged shapefile is created for each processed OSM layer.
    - Duplicate records are removed safely using geometry-aware comparison.

Usage examples:

  # Basic usage (only required parameters):
  python merge_shp.py --base_dir "C:\...\maxar_raw" --output_dir "C:\...\merged"

  # Process specific categories:
  python merge_shp.py --base_dir "C:\...\maxar_raw" --output_dir "C:\...\merged" \
                      --categories buildings roads waterways

  # Exclude specific categories:
  python merge_shp.py --base_dir "C:\...\maxar_raw" --output_dir "C:\...\merged" \
                      --exclude places pofw
"""

import os
import argparse
import geopandas as gpd
import pandas as pd

# ════════════════════════════════════════════════════════════
#  ARGPARSE
# ════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(
    description="Scans OSM shapefile folders and merges shapefiles by category."
)

parser.add_argument(
    "--base_dir", required=True,
    help="Root directory to be scanned, including subfolders"
)
parser.add_argument(
    "--output_dir", required=True,
    help="Folder where the merged .shp files will be saved"
)
parser.add_argument(
    "--categories", nargs="*", default=None,
    help=(
        "Short names of the categories to be processed, separated by spaces. "
        "If not provided, all categories will be processed. "
        "Example: --categories buildings roads waterways"
    )
)
parser.add_argument(
    "--exclude", nargs="*", default=[],
    help=(
        "Short names of the categories to be excluded. "
        "Example: --exclude places pofw"
    )
)

args = parser.parse_args()

# ════════════════════════════════════════════════════════════
#  ALL categories
# ════════════════════════════════════════════════════════════

ALL_SHP_NAMES = [
    "gis_osm_buildings_a_free_1.shp",
    "gis_osm_landuse_a_free_1.shp",
    "gis_osm_natural_a_free_1.shp",
    "gis_osm_natural_free_1.shp",
    "gis_osm_places_a_free_1.shp",
    "gis_osm_places_free_1.shp",
    "gis_osm_pofw_a_free_1.shp",
    "gis_osm_pofw_free_1.shp",
    "gis_osm_pois_a_free_1.shp",
    "gis_osm_pois_free_1.shp",
    "gis_osm_railways_free_1.shp",
    "gis_osm_roads_free_1.shp",
    "gis_osm_traffic_a_free_1.shp",
    "gis_osm_traffic_free_1.shp",
    "gis_osm_transport_a_free_1.shp",
    "gis_osm_transport_free_1.shp",
    "gis_osm_water_a_free_1.shp",
    "gis_osm_waterways_free_1.shp",
    # Optional political boundary layers for raw/manual OSM downloads.
    # In the formatted workflow, political boundaries are usually merged into landuse.
    "gis_osm_political_boundaries_a_free_1.shp",
    "gis_osm_political_boundaries_free_1.shp",
]

# ── Apply Filter ─────────────────────────────────────────────────
if args.categories:
    # Convert provided short category names to full shapefile names
    # For example, "buildings" may match "gis_osm_buildings_a_free_1.shp"
    shp_names = [
        s for s in ALL_SHP_NAMES
        if any(cat.lower() in s.lower() for cat in args.categories)
    ]
    if not shp_names:
        print("[ERROR] The provided --categories did not match any shapefile.")
        print("Valid categories:", ", ".join(ALL_SHP_NAMES))
        exit(1)
else:
    shp_names = ALL_SHP_NAMES

if args.exclude:
    shp_names = [
        s for s in shp_names
        if not any(ex.lower() in s.lower() for ex in args.exclude)
    ]

# ════════════════════════════════════════════════════════════
#  Processing categories
# ════════════════════════════════════════════════════════════

base_dir   = args.base_dir
output_dir = args.output_dir

os.makedirs(output_dir, exist_ok=True)

print(f"Base directory  : {base_dir}")
print(f"Output     : {output_dir}")
print(f"Categories   : {len(shp_names)}")
if args.exclude:
    print(f"Excluded categories      : {args.exclude}")
print("-" * 55)

total_categories  = 0
total_features     = 0

for shp_name in shp_names:
    all_gdfs = []

    for root, dirs, files in os.walk(base_dir):
        if shp_name in files:
            full_path = os.path.join(root, shp_name)
            try:
                temp_gdf = gpd.read_file(full_path)
                if temp_gdf is not None and not temp_gdf.empty:
                    all_gdfs.append(temp_gdf)
            except Exception as e:
                print(f"  [ERROR] Could not read: {full_path}  –  {e}")

    if not all_gdfs:
        print(f"Skipped : {shp_name}  (not found in any directory)")
        continue

    print(f"Merging: {shp_name}  ({len(all_gdfs)} source folders)")

    try:
        merged_df     = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs=all_gdfs[0].crs)
        initial_count = len(merged_df)

        if "osm_id" in merged_df.columns:
            merged_df["osm_id"] = merged_df["osm_id"].astype(str)

            merged_df["_geom_wkb"] = merged_df.geometry.apply(
                lambda g: g.wkb_hex if g is not None else ""
            )

            merged_df = merged_df.drop_duplicates(
                subset=["osm_id", "_geom_wkb"],
                keep="first"
            )

            merged_df = merged_df.drop(columns=["_geom_wkb"])
        else:
            merged_df["_geom_wkb"] = merged_df.geometry.apply(
                lambda g: g.wkb_hex if g is not None else ""
            )

            merged_df = merged_df.drop_duplicates(
                subset=["_geom_wkb"],
                keep="first"
            )

            merged_df = merged_df.drop(columns=["_geom_wkb"])

        final_count   = len(merged_df)
        removed_count = initial_count - final_count

        save_path = os.path.join(output_dir, shp_name)
        merged_df.to_file(save_path, encoding="utf-8")

        print(f"   -> OK: {final_count:,} features  (duplicates removed: {removed_count:,})")
        total_categories += 1
        total_features    += final_count

    except Exception as e:
        print(f"   -> [ERROR] Merge failed: {e}")

print("-" * 55)
print(f"Completed: {total_categories} categories  |  total {total_features:,} features")
print(f"Output    : {output_dir}")