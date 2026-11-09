"""

osm_style_processor.py – Headless QGIS OSM Tile Generator

This script runs in a QGIS Python environment and generates paired raster
and OSM mask tiles. It loads predefined OSM shapefile layers, applies their
corresponding QML style files, and renders the styled OSM layers over the
spatial extent of each raster tile.

For each input raster, the script creates:
1. Raster image tiles extracted directly from the source raster.
2. OSM mask tiles rendered from styled OSM vector layers using QGIS.

The output raster tiles and OSM mask tiles have the same tile size, spatial
extent, geotransform, and projection, so they can be used as paired datasets
for image processing, semantic segmentation, or machine learning workflows.

Input parameters:
--shp_dir:
Folder containing the OSM shapefile (.shp) layers.

--qml_dir:
    Folder containing the QGIS style (.qml) files.
    The QML filenames must match the corresponding shapefile base names.

--output:
    Main output directory where the generated tiles will be saved.

--raster_dir:
    Folder containing raster files to be processed.
    Tested raster extensions: .tif and .jp2.
    Either --raster_dir or --raster_files must be provided.

--raster_files:
    Full paths of individual raster files to be processed.
    Multiple raster files can be provided, separated by spaces.
    Either --raster_dir or --raster_files must be provided.

--tile_size:
    Tile size in pixels.
    Default value: 256.

--dpi:
    Rendering DPI used for QGIS map rendering.
    Default value: 96.

--exclude:
    Optional layer-name keywords to exclude from OSM mask rendering.
    Example: --exclude places pofw

Outputs:
For each input raster, a separate output folder is created using the
raster filename. Inside this folder, two subfolders are generated:

    raster/
        Contains raster image tiles.
        Example:
        raster_name_patch_0.tif

    osm/
        Contains rendered OSM mask tiles.
        Example:
        raster_name_mask_0.tif

The raster and OSM mask tiles are spatially aligned and use the same
tile index.


osm_style_processor.py
─────────────────────────────────────────────────────────────
Usage examples:

  # Using a raster folder:
  run.bat --shp_dir "C:\...\shp" --qml_dir "C:\...\styles" \
               --raster_dir "C:\...\rasters" --output "C:\...\tiles"

  # Using individual raster files:
  run.bat --shp_dir "C:\...\shp" --qml_dir "C:\...\styles" \
               --raster_files "C:\...\r1.tif" "C:\...\r2.tif" --output "C:\...\tiles"

  # Using optional parameters:
  --tile_size 512 --dpi 150 --exclude places pofw
"""

import os
import sys
import argparse

# ════════════════════════════════════════════════════════════
#  ARGPARSE
# ════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(
    description="Headless QGIS: apply OSM styles and generate raster/OSM tiles"
)

# Zorunlu
parser.add_argument("--shp_dir",  required=True,
                    help="Folder containing the OSM shapefile (.shp) files")
parser.add_argument("--qml_dir",  required=True,
                    help="Folder containing QGIS style files (.qml)")
parser.add_argument("--output",   required=True,
                    help="Main output folder for generated tiles")

# Raster input source (one of the two options is required)
raster_group = parser.add_mutually_exclusive_group(required=True)
raster_group.add_argument("--raster_dir",
                           help="Folder containing raster files (tif/tiff/img/ecw/jp2)")
raster_group.add_argument("--raster_files", nargs="+",
                           help="Full paths of raster files, separated by spaces")

# Opsiyonel
parser.add_argument("--tile_size", type=int, default=256,
                    help="Tile size in pixels (default: 256)")
parser.add_argument("--dpi",       type=int, default=96,
                    help="Rendering DPI (default: 96)")
parser.add_argument("--exclude",   nargs="*", default=[],
                    help="Layer-name substrings to exclude from OSM tile rendering")

args = parser.parse_args()

SHP_DIR          = args.shp_dir
QML_DIR          = args.qml_dir
OUTPUT_BASE      = args.output
TILE_SIZE_PX     = args.tile_size
DPI              = args.dpi
EXCLUDE_KEYWORDS = [k.lower() for k in args.exclude]

if args.raster_dir:
    exts = (".tif", ".tiff", ".img", ".ecw", ".jp2")
    raster_paths = [
        os.path.join(args.raster_dir, f)
        for f in os.listdir(args.raster_dir)
        if f.lower().endswith(exts)
    ]
else:
    raster_paths = [p for p in args.raster_files if os.path.exists(p)]

# ════════════════════════════════════════════════════════════
#  QGIS Initialization
# ════════════════════════════════════════════════════════════

from qgis.core import (
    QgsApplication,
    QgsProject,
    QgsMapSettings,
    QgsMapRendererCustomPainterJob,
    QgsRectangle,
    QgsRasterLayer,
    QgsVectorLayer,
)
from PyQt5.QtGui  import QImage, QPainter, QColor
from PyQt5.QtCore import QSize
from osgeo import gdal
import numpy as np

# Make the shared GenCP_data_preparation package importable when this file is
# run as a standalone script (repo root is three levels up from this module:
# genCP/GenCP_data_preparation/VHR/<this file>).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from GenCP_data_preparation.common import raster_io as _raster_io
from GenCP_data_preparation.common import tiling as _tiling

qgs = QgsApplication([], False)
qgs.initQgis()
print("[OK] QGIS started in headless mode.")
print(f"     shp_dir   : {SHP_DIR}")
print(f"     qml_dir   : {QML_DIR}")
print(f"     output    : {OUTPUT_BASE}")
print(f"     tile_size : {TILE_SIZE_PX}px  |  dpi: {DPI}")
print(f"     raster    : {len(raster_paths)} files")
print(f"     exclude   : {EXCLUDE_KEYWORDS or '-'}")

# ════════════════════════════════════════════════════════════
#  OSM layer list (top-to-bottom order in the layer panel)
# ════════════════════════════════════════════════════════════
BASE_NAMES = [
    "gis_osm_transport_free_1",
    "gis_osm_transport_a_free_1",
    "gis_osm_traffic_free_1",
    "gis_osm_traffic_a_free_1",
    "gis_osm_pois_free_1",
    "gis_osm_pofw_free_1",
    "gis_osm_places_free_1",
    "gis_osm_natural_free_1",
    
    "gis_osm_roads_free_1",
    "gis_osm_railways_free_1",
    "gis_osm_buildings_a_free_1",
    
    "gis_osm_water_a_free_1",
    "gis_osm_waterways_free_1",

    "gis_osm_pois_a_free_1",
    "gis_osm_pofw_a_free_1",
    
    "gis_osm_natural_a_free_1",
    "gis_osm_landuse_a_free_1",
]

# ════════════════════════════════════════════════════════════
#  SECTION 1 – Load OSM Layers and Apply Styles
# ════════════════════════════════════════════════════════════

project = QgsProject.instance()

print("\n" + "="*60)
print("SECTION 1: Loading OSM layers...")
print("="*60)

for name in BASE_NAMES:
    shp_path = os.path.join(SHP_DIR, f"{name}.shp")
    qml_path = os.path.join(QML_DIR, f"{name}.qml")

    if not os.path.exists(shp_path):
        print(f"  [SKIP] SHP not found:: {shp_path}")
        continue
    if not os.path.exists(qml_path):
        print(f"  [SKIP] QML not found: {qml_path}")
        continue

    layer = QgsVectorLayer(shp_path, name, "ogr")
    if not layer.isValid():
        print(f"  [ERROR] Invalid layer: {name}")
        continue

    msg = layer.loadNamedStyle(qml_path)
    print(f"  [OK]  {name}  |  loadNamedStyle: {repr(msg)}")
    layer.triggerRepaint()
    project.addMapLayer(layer)

# Enable symbol levels for the landuse layer so that its internal symbol
# rendering order is correctly preserved during tile generation

lu = project.mapLayersByName("gis_osm_landuse_a_free_1")
if lu:
    r = lu[0].renderer()
    if hasattr(r, "setUsingSymbolLevels"):
        r.setUsingSymbolLevels(True)
    lu[0].triggerRepaint()
    print("Symbol levels forced: gis_osm_landuse_a_free_1")

# Create an ordered layer list for rendering
def get_ordered_layers():
    ordered = []
    for name in BASE_NAMES:
        if any(kw in name.lower() for kw in EXCLUDE_KEYWORDS):
            print(f"  [EXCLUDED] {name}")
            continue
        matches = project.mapLayersByName(name)
        if matches:
            ordered.append(matches[0])
        else:
            print(f"  [WARNING] Layer not found in the project: {name}")
    return ordered

ordered_vector_layers = get_ordered_layers()
print(f"\Total {len(ordered_vector_layers)} layers are ready for tile generation.")

# ════════════════════════════════════════════════════════════
#  Utility Functions
# ════════════════════════════════════════════════════════════

def qimage_to_rgb_array(img: QImage):
    img = img.convertToFormat(QImage.Format_ARGB32)
    ptr = img.bits()
    ptr.setsize(img.byteCount())
    arr = np.frombuffer(ptr, np.uint8).reshape((img.height(), img.width(), 4))
    return np.stack([arr[..., 2], arr[..., 1], arr[..., 0]], axis=0)

def sanitize_name(name: str) -> str:
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, "_")
    return name.strip().rstrip(".")

def gdal_dtype(arr: np.ndarray):
    # Delegates to the shared helper.
    return _raster_io.gdal_dtype(arr)

def write_geotiff(path, data, tile_gt, proj_wkt, n_bands, dtype=None):
    # Delegates to the shared helper; width/height are derived from ``data``.
    _raster_io.write_geotiff(path, data, tile_gt, proj_wkt, n_bands=n_bands, dtype=dtype)

# ════════════════════════════════════════════════════════════
#  SECTION 2 – Raster Tile Generation
# ════════════════════════════════════════════════════════════

print("\n" + "="*60)
print(f"SECTION 2: Tile generation  ({len(raster_paths)} rasters)")
print("="*60)

if not raster_paths:
    print("[WARNING] No raster files found to process; tile generation will be skipped.")

total_tiles = 0

for raster_path in raster_paths:
    base_name = os.path.splitext(os.path.basename(raster_path))[0]
    out_root  = os.path.join(OUTPUT_BASE, sanitize_name(base_name))
    ras_dir   = os.path.join(out_root, "raster")
    osm_dir   = os.path.join(out_root, "osm")
    os.makedirs(ras_dir, exist_ok=True)
    os.makedirs(osm_dir, exist_ok=True)

    rlayer = QgsRasterLayer(raster_path, base_name)
    if not rlayer.isValid():
        print(f"\n  [ERROR] Invalid QgsRasterLayer: {base_name}")
        continue

    src = gdal.Open(raster_path)
    if src is None:
        print(f"\n  [ERROR] Could not open raster with GDAL: {raster_path}")
        continue

    gt  = src.GetGeoTransform()
    wkt = src.GetProjection()
    ox, pw, rx = gt[0], gt[1], gt[2]
    oy, ry, ph = gt[3], gt[4], gt[5]
    W, H = src.RasterXSize, src.RasterYSize

    nc = W // TILE_SIZE_PX
    nr = H // TILE_SIZE_PX
    print(f"\n  {base_name}  ({W}x{H} px)  ->  {nc}x{nr} tile grid")

    if nc == 0 or nr == 0:
        print("  [SKIP] Tile size is larger than the raster dimensions.")
        src = None
        continue

    count = 0

    for i in range(nr):
        for j in range(nc):
            px = j * TILE_SIZE_PX
            py = i * TILE_SIZE_PX

            if px + TILE_SIZE_PX > W or py + TILE_SIZE_PX > H:
                continue

            tile_gt = _tiling.tile_geotransform(gt, px, py)
            tox, toy = tile_gt[0], tile_gt[3]
            idx     = i * nc + j

            # A) Raster tile
            data = src.ReadAsArray(px, py, TILE_SIZE_PX, TILE_SIZE_PX)
            if data is None:
                continue
            if data.ndim == 2:
                data = data[np.newaxis, ...]

            write_geotiff(
                os.path.join(ras_dir, f"{base_name}_patch_{idx}.tif"),
                data, tile_gt, wkt, data.shape[0]
            )

            # B) OSM mask tile
            x2   = tox + TILE_SIZE_PX * pw
            y2   = toy + TILE_SIZE_PX * ph
            rect = QgsRectangle(min(tox, x2), min(toy, y2), max(tox, x2), max(toy, y2))

            ms = QgsMapSettings()
            ms.setDestinationCrs(rlayer.crs())
            ms.setLayers(ordered_vector_layers)
            ms.setExtent(rect)
            ms.setOutputDpi(DPI)
            ms.setOutputSize(QSize(TILE_SIZE_PX, TILE_SIZE_PX))

            # Set the OSM mask background color to black
            ms.setBackgroundColor(QColor(0, 0, 0, 255))

            img = QImage(TILE_SIZE_PX, TILE_SIZE_PX, QImage.Format_ARGB32)
                        
            # Initialize the image with a black background
            img.fill(QColor(0, 0, 0, 255))
                        
            painter = QPainter(img)
            job = QgsMapRendererCustomPainterJob(ms, painter)
            job.start()
            job.waitForFinished()
            painter.end()

            rgb = qimage_to_rgb_array(img)
            write_geotiff(
                os.path.join(osm_dir, f"{base_name}_mask_{idx}.tif"),
                rgb, tile_gt, wkt, 3, dtype=gdal.GDT_Byte
            )

            count += 1
            if count % 200 == 0:
                print(f"    ... {count} tiles processed.")

    print(f"  Completed: {count} tile  ->  {out_root}")
    total_tiles += count
    src = None

# ════════════════════════════════════════════════════════════
#  Finalization
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"DONE.  Total tile: {total_tiles}")
print(f"Output: {OUTPUT_BASE}")
print("="*60)

#qgs.exitQgis()
sys.exit(0)
