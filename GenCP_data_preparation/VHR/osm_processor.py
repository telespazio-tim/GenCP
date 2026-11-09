#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
osm_processor.py - OSM Layer Downloader for Raster or Vector Boundaries

This script downloads OSM data based on the spatial extent of a raster or vector
input file. It automatically detects the input type, computes its bounding box,
converts the extent to WGS84, and downloads OSM features using OSMnx and the
Overpass API.

All required OSM tags are combined into a single Overpass request. After the
combined download is completed, the returned GeoDataFrame is filtered locally
according to the predefined layer definitions. Each layer is then filtered by
geometry type, converted back to the input CRS, cleaned for shapefile
compatibility, and saved as a separate shapefile.

Download strategy:
- First, the script tries the full bounding box without splitting.
- If no response is received within 180 seconds, the same full query is tried
  one more time.
- If the second full-bbox attempt also fails, the bounding box is split into
  smaller parts and downloaded piece by piece.
- OSMnx caching is enabled to reuse previous Overpass responses.

Input parameters:
    --input / -i:
        Raster or vector file used to define the download area.
        Supported examples: .tif, .tiff, .vrt, .shp, .geojson, .gpkg, .kml.

    --output / -o:
        Folder where the downloaded OSM shapefiles will be saved.

    --buffer:
        Buffer distance added around the input bounding box, in meters.
        Default: 0.

    --layers:
        Optional short names of OSM layers to be processed.
        If not provided, all predefined layers are processed.
        Example: --layers buildings roads waterways

Output:
    - One shapefile is created for each processed OSM layer.
"""

import os
import sys
import time
import argparse
import pandas as pd
import geopandas as gpd
import rasterio
import osmnx as ox

from rasterio.warp import transform_bounds
from rasterio.crs import CRS
from shapely.geometry import box

# Make the shared GenCP_data_preparation package importable when this file is
# run as a standalone script (repo root is three levels up from this module:
# genCP/GenCP_data_preparation/VHR/<this file>).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from GenCP_data_preparation.common import geo as _geo
from GenCP_data_preparation.common import osm_download as _osm_dl


FULL_QUERY_TIMEOUT = 180
SPLIT_QUERY_TIMEOUT = 180
SPLIT_N = 2
OVERPASS_MAXSIZE = 1073741824


LAYER_SPECS = {
    "gis_osm_buildings_a_free_1": {
        "tags": {"building": True},
        "geometry_types": ["Polygon", "MultiPolygon"],
    },
    "gis_osm_landuse_a_free_1": {
        "tags": {"landuse": True},
        "geometry_types": ["Polygon", "MultiPolygon"],
    },
    "gis_osm_natural_a_free_1": {
        "tags": {"natural": True},
        "geometry_types": ["Polygon", "MultiPolygon"],
    },
    "gis_osm_natural_free_1": {
    "tags": {"natural": True},
    "geometry_types": ["Point", "MultiPoint"],
    },
    "gis_osm_places_a_free_1": {
        "tags": {"place": True},
        "geometry_types": ["Polygon", "MultiPolygon"],
    },
    "gis_osm_places_free_1": {
        "tags": {"place": True},
        "geometry_types": ["Point", "MultiPoint"],
    },
    "gis_osm_pofw_a_free_1": {
        "tags": {"amenity": "place_of_worship"},
        "geometry_types": ["Polygon", "MultiPolygon"],
    },
    "gis_osm_pofw_free_1": {
        "tags": {"amenity": "place_of_worship"},
        "geometry_types": ["Point", "MultiPoint"],
    },
    "gis_osm_pois_a_free_1": {
        "tags": {
            "amenity": True,
            "shop": True,
            "tourism": True,
            "leisure": True,
            "office": True,
            "healthcare": True,
            "historic": True,
            "man_made": True,
            "emergency": True,
            "craft": True,
        },
        "geometry_types": ["Polygon", "MultiPolygon"],
    },
    "gis_osm_pois_free_1": {
    "tags": {
        "amenity": True,
        "shop": True,
        "tourism": True,
        "leisure": True,
        "office": True,
        "healthcare": True,
        "historic": True,
        "man_made": True,
        "emergency": True,
        "craft": True,
    },
    "geometry_types": ["Point", "MultiPoint"],
    },
    "gis_osm_railways_free_1": {
        "tags": {"railway": True},
        "geometry_types": ["LineString", "MultiLineString"],
    },
    "gis_osm_roads_free_1": {
        "tags": {"highway": True},
        "geometry_types": ["LineString", "MultiLineString"],
    },
    "gis_osm_traffic_a_free_1": {
        "tags": {
            "amenity": ["parking", "fuel"],
            "highway": ["services", "rest_area"],
            "railway": ["station", "halt"],
        },
        "geometry_types": ["Polygon", "MultiPolygon"],
    },
    "gis_osm_traffic_free_1": {
        "tags": {
            "amenity": ["parking", "fuel"],
            "highway": ["bus_stop", "motorway_junction", "traffic_signals"],
            "railway": ["station", "halt", "tram_stop"],
        },
        "geometry_types": ["Point", "MultiPoint", "LineString", "MultiLineString"],
    },
    "gis_osm_transport_a_free_1": {
        "tags": {
            "amenity": ["bus_station", "ferry_terminal"],
            "aeroway": ["aerodrome", "terminal"],
            "landuse": ["railway"],
        },
        "geometry_types": ["Polygon", "MultiPolygon"],
    },
    "gis_osm_transport_free_1": {
        "tags": {
            "amenity": ["bus_station", "ferry_terminal"],
            "aeroway": ["aerodrome", "terminal", "gate"],
            "public_transport": True,
        },
        "geometry_types": ["Point", "MultiPoint", "LineString", "MultiLineString"],
    },
    "gis_osm_water_a_free_1": {
        "tags": {
            "natural": "water",
            "landuse": "reservoir",
            "waterway": "riverbank",
            "water": True,
        },
        "geometry_types": ["Polygon", "MultiPolygon"],
    },
    "gis_osm_waterways_free_1": {
        "tags": {"waterway": True},
        "geometry_types": ["LineString", "MultiLineString"],
    },
    "gis_osm_political_boundaries_a_free_1": {
    "tags": {"boundary": ["political", "administrative"]},
    "geometry_types": ["Polygon", "MultiPolygon"],
    },
        
    # "gis_osm_political_boundaries_a_free_1": 
    # "tags": {"boundary": "political"},
    # "geometry_types": ["Polygon", "MultiPolygon"],
    # },
}


RASTER_EXTS = {".tif", ".tiff", ".vrt", ".img", ".ecw", ".jp2"}
VECTOR_EXTS = {".shp", ".geojson", ".kml", ".gpkg", ".json"}


def compute_bbox_from_raster(input_path, buffer_meters=0):
    # Delegates to the shared helper; returns ((west, south, east, north), crs).
    return _geo.bbox_from_raster(input_path, buffer_meters=buffer_meters)


def compute_bbox_from_vector(input_path, buffer_meters=0):
    # Delegates to the shared helper; returns ((west, south, east, north), crs).
    return _geo.bbox_from_vector(input_path, buffer_meters=buffer_meters)


def get_bbox(input_path, buffer_meters=0):
    ext = os.path.splitext(input_path)[1].lower()

    if ext in RASTER_EXTS:
        print(f"[INFO] Input type: RASTER  ->  {os.path.basename(input_path)}", flush=True)
        return compute_bbox_from_raster(input_path, buffer_meters)

    if ext in VECTOR_EXTS:
        print(f"[INFO] Input type: VECTOR  ->  {os.path.basename(input_path)}", flush=True)
        return compute_bbox_from_vector(input_path, buffer_meters)

    raise ValueError(f"Unsupported file format: {ext}")


def configure_overpass(timeout=180, maxsize=1073741824):
    ox.settings.log_console = True
    ox.settings.use_cache = True
    ox.settings.cache_folder = ".\\cache"

    ox.settings.overpass_url = "https://overpass-api.de/api"

    ox.settings.http_user_agent = "VHR_OSM_STYLED_DATASET_PREP/1.0"
    ox.settings.http_referer = "https://overpass-turbo.eu/"
    ox.settings.http_accept_language = "en"

    try:
        ox.settings.requests_timeout = timeout
    except Exception:
        pass

    try:
        ox.settings.timeout = timeout
    except Exception:
        pass

    try:
        ox.settings.overpass_memory = maxsize
    except Exception:
        pass

    ox.settings.overpass_settings = "[out:json][timeout:{timeout}]{maxsize}"

    print("Overpass URL:", ox.settings.overpass_url, flush=True)
    print("Cache folder:", ox.settings.cache_folder, flush=True)
    print("Timeout:", timeout, flush=True)


def split_bbox_wgs84(bbox_wgs84, n=2):
    # Delegates to the shared helper (canonical (west, south, east, north)).
    return _osm_dl.split_bbox(bbox_wgs84, n)


def build_combined_tags(layer_specs):
    # Delegates to the shared helper; merges each layer spec's tag dict.
    return _osm_dl.combine_tags(spec["tags"] for spec in layer_specs.values())


def is_no_data_error(error):
    msg = str(error)

    no_data_patterns = [
        "No data elements",
        "Found no graph nodes",
        "InsufficientResponseError",
    ]

    return any(pattern in msg for pattern in no_data_patterns)


def _download_once(bbox_wgs84, tags):
    west, south, east, north = bbox_wgs84

    print(
        f"    -> Overpass: "
        f"N={north:.5f} S={south:.5f} E={east:.5f} W={west:.5f}",
        flush=True,
    )

    gdf = _osm_dl.download_features((west, south, east, north), tags)

    print(f"    -> {len(gdf)} features returned", flush=True)

    return gdf


def download_combined_osm_data(bbox_wgs84, combined_tags):
    configure_overpass(
        timeout=FULL_QUERY_TIMEOUT,
        maxsize=OVERPASS_MAXSIZE,
    )

    last_error = None

    for attempt in range(1, 3):
        try:
            print(f"[INFO] Full bbox query attempt {attempt}/2", flush=True)

            gdf = _download_once(bbox_wgs84, combined_tags)

            if gdf is None or gdf.empty:
                raise RuntimeError("Empty Overpass response")

            return gdf

        except Exception as e:
            last_error = e

            if is_no_data_error(e):
                print(f"[WARN] No-data-like response received, retry will continue: {e}", flush=True)
            else:
                print(f"[WARN] Full bbox query attempt {attempt}/2 failed: {e}", flush=True)

            if attempt < 2:
                time.sleep(5)

    print("[INFO] Switching to split download.", flush=True)
    print(f"[WARN] Last full-bbox error: {last_error}", flush=True)

    configure_overpass(
        timeout=SPLIT_QUERY_TIMEOUT,
        maxsize=OVERPASS_MAXSIZE,
    )

    all_gdfs = []
    parts = split_bbox_wgs84(bbox_wgs84, n=SPLIT_N)

    print(f"[INFO] BBox was split into {len(parts)} parts.", flush=True)

    for k, part_bbox in enumerate(parts, 1):
        print(f"[PART {k}/{len(parts)}]", flush=True)

        part_success = False

        for part_attempt in range(1, 3):
            try:
                print(f"    Part attempt {part_attempt}/2", flush=True)

                gdf_part = _download_once(part_bbox, combined_tags)

                if gdf_part is not None and not gdf_part.empty:
                    all_gdfs.append(gdf_part)
                    part_success = True
                    print(f"    [OK] Part downloaded: {len(gdf_part)} features", flush=True)
                    break

                print("    [WARN] Empty response for this part.", flush=True)

            except Exception as e:
                print(
                    f"    [WARN] Split part attempt {part_attempt}/2 failed: {e}",
                    flush=True
                )

            time.sleep(3)

        if not part_success:
            print("    [WARN] This split part could not be downloaded.", flush=True)

    if not all_gdfs:
        print("[WARN] No data downloaded after split query.", flush=True)
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    merged = gpd.GeoDataFrame(
        pd.concat(all_gdfs, ignore_index=False),
        crs=all_gdfs[0].crs,
    )

    try:
        merged = merged[~merged.index.duplicated(keep="first")]
    except Exception:
        pass

    print(f"[OK] Split download completed. Total features: {len(merged)}", flush=True)

    return merged


def tag_match(gdf, key, value):
    if key not in gdf.columns:
        return pd.Series(False, index=gdf.index)

    col = gdf[key]

    if value is True:
        return col.notna() & (col.astype(str) != "")

    if isinstance(value, list):
        return col.isin(value)

    return col == value


def filter_by_tags(gdf, tags):
    if gdf.empty:
        return gdf

    mask = pd.Series(False, index=gdf.index)

    for key, value in tags.items():
        mask = mask | tag_match(gdf, key, value)

    return gdf[mask].copy()


def filter_geom_types(gdf, geometry_types):
    if not geometry_types:
        return gdf

    return gdf[gdf.geometry.geom_type.isin(geometry_types)].copy()


def ensure_crs(gdf, target_crs):
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)

    return gdf.to_crs(target_crs)


def clean_list_columns(gdf):
    for col in gdf.columns:
        if gdf[col].apply(lambda x: isinstance(x, (list, dict, tuple, set))).any():
            gdf[col] = gdf[col].astype(str)

    return gdf


# Columns that should be preserved in the raw OSM shapefiles.
# These are the columns needed by the formatter and QML workflow.
RAW_KEEP_COLUMNS = [
    "osm_id",
    "name",

    # Buildings
    "building",

    # Roads
    "highway", "ref", "oneway", "maxspeed", "layer", "bridge", "tunnel",

    # Railways
    "railway",

    # Natural / water / landuse
    "natural",
    "waterway", "water",
    "landuse",

    # Places
    "place", "population",

    # POIs
    "amenity",
    "shop",
    "tourism",
    "leisure",
    "office",
    "healthcare",
    "historic",
    "man_made",
    "emergency",
    "craft",

    # Transport
    "aeroway",
    "public_transport",

    # Boundaries / religion
    "boundary",
    "religion",

    # Other useful attributes
    "width",
]


def _value_to_shp_string(value):
    """Convert one attribute value to a shapefile-safe string."""
    if value is None:
        return ""

    if isinstance(value, (list, dict, tuple, set)):
        return str(value)[:254]

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value)[:254]


def prepare_for_shapefile(gdf):
    """
    Keep only useful OSM columns and convert attributes to shapefile-safe values.

    Combined OSMnx outputs may contain many object/list/dict columns. ESRI
    Shapefile is strict about field types and field lengths. This function
    reduces the raw output to the columns needed by the downstream formatter.
    """
    gdf = gdf.copy()

    # OSMnx often stores the id in the GeoDataFrame index instead of a column.
    if "osm_id" not in gdf.columns:
        if "osmid" in gdf.columns:
            gdf["osm_id"] = gdf["osmid"].astype(str)
        elif isinstance(gdf.index, pd.MultiIndex):
            if "osmid" in gdf.index.names:
                gdf["osm_id"] = gdf.index.get_level_values("osmid").astype(str)
            else:
                gdf["osm_id"] = gdf.index.get_level_values(-1).astype(str)
        else:
            gdf["osm_id"] = gdf.index.astype(str)

    # Do not keep an "osmid" field in the output shapefile.
    # OSMnx often has osmid in the index as well; keeping both can trigger
    # "cannot insert osmid, already exists" during GeoPandas/Fiona writing.
    if "osmid" in gdf.columns:
        gdf = gdf.drop(columns=["osmid"])

    keep_cols = [col for col in RAW_KEEP_COLUMNS if col in gdf.columns]
    if "geometry" not in keep_cols:
        keep_cols.append("geometry")

    gdf = gdf[keep_cols].copy()

    # Remove MultiIndex/index names before writing so no index field is inserted.
    gdf = gdf.reset_index(drop=True)
    gdf.index.name = None

    gdf = clean_list_columns(gdf)

    for col in gdf.columns:
        if col == "geometry":
            continue
        gdf[col] = gdf[col].apply(_value_to_shp_string)

    return gdf


def process_area(input_path, out_dir, buffer_meters=0, layer_filter=None):
    bbox_wgs84, target_crs = get_bbox(input_path, buffer_meters)

    print(f"[INFO] Target CRS : {target_crs}", flush=True)
    print(
        f"[INFO] BBox (WGS84): "
        f"W={bbox_wgs84[0]:.5f} S={bbox_wgs84[1]:.5f} "
        f"E={bbox_wgs84[2]:.5f} N={bbox_wgs84[3]:.5f}",
        flush=True,
    )

    if buffer_meters:
        print(f"[INFO] Buffer    : {buffer_meters} m", flush=True)

    os.makedirs(out_dir, exist_ok=True)

    if layer_filter:
        specs = {
            k: v for k, v in LAYER_SPECS.items()
            if any(f.lower() in k.lower() for f in layer_filter)
        }

        if not specs:
            print("[ERROR] The --layers argument did not match any available layer.", flush=True)
            return
    else:
        specs = LAYER_SPECS

    print(f"[INFO] Number of layers to be processed: {len(specs)}", flush=True)

    combined_tags = build_combined_tags(specs)

    print("[INFO] Combined Overpass query will be used.", flush=True)
    print(f"[INFO] Combined tags: {combined_tags}", flush=True)

    try:
        full_gdf = download_combined_osm_data(bbox_wgs84, combined_tags)
    except Exception as e:
        print(f"[ERROR] Combined OSM download failed: {e}", flush=True)
        raise

    if full_gdf.empty:
        raise RuntimeError("No OSM data downloaded for this area.")

    print(f"[OK] Combined OSM download completed. Total features: {len(full_gdf)}", flush=True)
    print("[DEBUG] Full geometry types:", flush=True)
    print(full_gdf.geometry.geom_type.value_counts(), flush=True)
    print("[DEBUG] Available columns:", flush=True)
    print(list(full_gdf.columns), flush=True)

    written_count = 0
    write_errors = []

    for shp_basename, spec in specs.items():
        print(f"[INFO] Layer: {shp_basename}", flush=True)

        gdf = filter_by_tags(full_gdf, spec["tags"])

        if gdf.empty:
            print(f"[INFO] {shp_basename} -> empty layer, skipped.", flush=True)
            continue

        gdf = filter_geom_types(gdf, spec.get("geometry_types"))

        if gdf.empty:
            print(f"[INFO] {shp_basename} -> empty after geometry filtering, skipped.", flush=True)
            continue

        gdf = ensure_crs(gdf, target_crs)
        gdf = prepare_for_shapefile(gdf)

        out_path = os.path.join(out_dir, f"{shp_basename}.shp")

        try:
            gdf.to_file(out_path, encoding="utf-8", index=False)
            print(f"[OK]  {out_path}  (n={len(gdf)})", flush=True)
            written_count += 1
        except Exception as e:
            write_errors.append((shp_basename, str(e)))
            print(f"[WARN] {shp_basename} write error: {e}", flush=True)

    if written_count == 0:
        if write_errors:
            print("[ERROR] Write errors were encountered:", flush=True)
            for layer_name, err in write_errors[:10]:
                print(f"  - {layer_name}: {err}", flush=True)
        raise RuntimeError("No non-empty OSM shapefile was written for this input.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downloads OSM layers based on raster or vector boundaries."
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input file (.tif/.tiff/.vrt/.shp/.geojson/.gpkg/.kml)",
    )

    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Folder where the output shapefiles will be saved.",
    )

    parser.add_argument(
        "--buffer",
        type=float,
        default=0,
        help="Buffer to be added to the bbox, in meters (default: 0)",
    )

    parser.add_argument(
        "--layers",
        nargs="*",
        default=None,
        help=(
            "Short names of the layers to be processed (default: all). "
            "Example: --layers buildings roads waterways"
        ),
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}", flush=True)
        exit(1)

    process_area(
        input_path=args.input,
        out_dir=args.output,
        buffer_meters=args.buffer,
        layer_filter=args.layers,
    )