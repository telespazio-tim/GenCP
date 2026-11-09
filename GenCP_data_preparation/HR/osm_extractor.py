"""OSM data extraction and rasterization for HR data preparation."""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import geopandas as gpd
import geojson
import numpy as np
import osmnx as ox
import requests
from osmnx._errors import InsufficientResponseError, ResponseStatusCodeError
import pyproj
import rasterio
import rasterio.transform
from osgeo import gdal, ogr, osr
from PIL import ImageColor
from pyproj import Transformer
from shapely.geometry import LineString, box
from shapely.ops import transform

from ..common.geo import bbox_from_raster
from ..common.osm_download import download_features
from . import config as hr_config
from .config import BBOX_BUFFER_METERS, DEFAULT_HIGHWAY_WIDTH, RASTERIZE_UPSAMPLE_FACTOR
from .geo_utils import get_stem, validate_file_exists

logger = logging.getLogger(__name__)


def extract_bbox_from_image(image_path: str) -> tuple:
    """Extract the image extent as a canonical EPSG:4326 bounding box.

    Delegates to :func:`GenCP_data_preparation.common.geo.bbox_from_raster`,
    buffered by ``BBOX_BUFFER_METERS`` in the image CRS.

    Note: previous versions returned ``(north, south, west, east)`` and buffered
    only two edges; this now returns the canonical ``(west, south, east, north)``
    order (as osmnx 2.x expects) with a symmetric buffer.

    Args:
        image_path: Path to input image.

    Returns:
        Bounding box ``(west, south, east, north)`` in EPSG:4326.
    """
    validate_file_exists(image_path, "input image")
    bbox, _ = bbox_from_raster(image_path, buffer_meters=BBOX_BUFFER_METERS)
    return bbox


def _flatten_list_columns(gdf) -> None:
    """Flatten columns that contain list values into comma-separated strings.

    Modifies the GeoDataFrame in place.
    """
    for column in gdf.columns:
        if gdf[column].apply(lambda x: isinstance(x, list)).any():
            gdf[column] = gdf[column].apply(
                lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x
            )


# ---------------------------------------------------------------------------
# In-memory OSM extraction and processing
# ---------------------------------------------------------------------------


def build_osm_query_tags(
    highway_colors: Optional[dict] = None,
    natural_colors: Optional[dict] = None,
    landuse_colors: Optional[dict] = None,
) -> dict:
    """Overpass tag filter restricted to what the HR rasterizer colours.

    Every building is kept (all are drawn brown), but highway / natural /
    landuse features are only requested for the values present in the colour
    dicts. Asking for ``key: True`` instead pulls every tree, crossing, bus
    stop... which inflates Overpass responses and only paints black (nodata)
    pixels that the CLC fill then has to recover.

    Args:
        highway_colors / natural_colors / landuse_colors: Colour mappings whose
            keys are the OSM values to request. Default to ``HR.config``.

    Returns:
        osmnx tag dict.
    """
    highway_colors = hr_config.highway_colors if highway_colors is None else highway_colors
    natural_colors = hr_config.natural_colors if natural_colors is None else natural_colors
    landuse_colors = hr_config.landuse_colors if landuse_colors is None else landuse_colors
    tags = {"building": True}
    for key, colors in (
        ("highway", highway_colors),
        ("natural", natural_colors),
        ("landuse", landuse_colors),
    ):
        if colors:
            tags[key] = sorted(colors)
    return tags


def extract_osm_data(bbox: tuple, image_name: str, tags: Optional[dict] = None) -> tuple:
    """Extract OSM data inside a bounding box via a single Overpass API call.

    Args:
        bbox: Bounding box for OSM data extraction limits.
        image_name: Image name (used for log messages).
        tags: osmnx tag filter. Defaults to :func:`build_osm_query_tags`.

    Returns:
        Tuple of (tags_data, roads_data) as in-memory GeoJSON dicts.

    Raises:
        InsufficientResponseError: If no OSM data is found at all.
    """
    try:
        gdf = download_features(
            bbox,
            tags=tags or build_osm_query_tags(),
        )
    except InsufficientResponseError:
        raise InsufficientResponseError(
            f"No OSM data at all for {image_name} — skipping tile"
        )

    _flatten_list_columns(gdf)

    empty_fc = {"type": "FeatureCollection", "features": []}

    # Split into roads vs tags based on highway column. When osmnx returns no
    # 'highway' column (a tile with no roads), fall back to an all-False mask —
    # `gdf.index < 0` raises TypeError on osmnx's (element_type, osmid) MultiIndex.
    has_highway = (
        gdf["highway"].notna() if "highway" in gdf.columns
        else np.zeros(len(gdf), dtype=bool)
    )
    gdf_roads = gdf[has_highway]
    gdf_tags = gdf[~has_highway]

    if gdf_roads.empty:
        roads_data = empty_fc.copy()
        logger.warning("No OSM highways found for %s", image_name)
    else:
        roads_data = json.loads(gdf_roads.to_json())

    if gdf_tags.empty:
        tags_data = empty_fc.copy()
        logger.warning("No OSM tags (building/landuse/natural) found for %s", image_name)
    else:
        tags_data = json.loads(gdf_tags.to_json())

    return tags_data, roads_data


def extract_osm_data_for_extent(
    extent_bbox: tuple,
    extent_name: str,
    tags: Optional[dict] = None,
) -> "gpd.GeoDataFrame":
    """Single Overpass call covering a union extent.

    Used by batched orchestrators that fetch OSM once for many chips and
    then subset spatially in memory, instead of issuing one Overpass call
    per chip.

    Args:
        extent_bbox: Bounding box passed to ``ox.features_from_bbox`` (same
            convention as ``extract_osm_data``).
        extent_name: Identifier used for log messages.
        tags: osmnx tag filter. Defaults to :func:`build_osm_query_tags`.

    Returns:
        GeoDataFrame in EPSG:4326 with all features. Empty (but valid) if
        Overpass returned nothing.
    """
    try:
        gdf = download_features(
            extent_bbox,
            tags=tags or build_osm_query_tags(),
        )
    except InsufficientResponseError:
        logger.warning("No OSM data at all for extent %s", extent_name)
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")

    _flatten_list_columns(gdf)
    return gdf


def _gdf_subset_to_geojson_dicts(
    gdf: "gpd.GeoDataFrame",
    chip_box_4326,
) -> tuple:
    """Spatial subset of a shared GeoDataFrame for one chip.

    Args:
        gdf: GeoDataFrame in EPSG:4326 (output of
            ``extract_osm_data_for_extent``).
        chip_box_4326: shapely geometry (typically a box) in EPSG:4326
            representing the chip's footprint.

    Returns:
        Tuple of (tags_data, roads_data) as in-memory GeoJSON dicts, with
        the same split semantics as ``extract_osm_data``.
    """
    empty_fc = {"type": "FeatureCollection", "features": []}

    if gdf.empty:
        return dict(empty_fc), dict(empty_fc)

    sindex = gdf.sindex
    candidate_idx = list(sindex.intersection(chip_box_4326.bounds))
    if not candidate_idx:
        return dict(empty_fc), dict(empty_fc)

    candidates = gdf.iloc[candidate_idx]
    chip_gdf = candidates[candidates.intersects(chip_box_4326)]

    if chip_gdf.empty:
        return dict(empty_fc), dict(empty_fc)

    has_highway = (
        chip_gdf["highway"].notna() if "highway" in chip_gdf.columns
        else np.zeros(len(chip_gdf), dtype=bool)
    )
    gdf_roads = chip_gdf[has_highway]
    gdf_tags = chip_gdf[~has_highway]

    roads_data = json.loads(gdf_roads.to_json()) if not gdf_roads.empty else dict(empty_fc)
    tags_data = json.loads(gdf_tags.to_json()) if not gdf_tags.empty else dict(empty_fc)

    return tags_data, roads_data


def set_highway_styles(
    roads_data: dict,
    highway_colors: dict,
    color_dict: dict,
    width: float,
) -> dict:
    """Compute highway styles from in-memory roads GeoJSON data.

    Args:
        roads_data: In-memory GeoJSON dict with highway features.
        highway_colors: Dictionary containing colors for highway features.
        color_dict: Dictionary containing color codes for color mapping.
        width: Buffer width to apply on highway features.

    Returns:
        Highway styles dictionary containing color and width for highway feature types.
    """
    highway_types = set(
        feature['properties']['highway']
        for feature in roads_data['features']
        if feature['properties'].get('highway')
    )

    highway_styles = {}
    width = float(width)
    for highway_type in highway_types:
        color = highway_colors.get(highway_type, '#000000')
        try:
            color = color_dict[color]
        except KeyError:
            color = '#000000'
        highway_styles[highway_type] = {'width': width, 'color': color}
    return highway_styles


def process_highway_data(roads_data: dict, highway_styles: dict) -> dict:
    """Set width and colors on highway features in memory.

    Args:
        roads_data: In-memory GeoJSON dict with highway features.
        highway_styles: Dictionary containing width and color per highway type.

    Returns:
        The mutated roads_data dict.
    """
    for feature in roads_data['features']:
        highway_type = feature['properties'].get('highway')
        if highway_type in highway_styles:
            style = highway_styles[highway_type]
            feature['properties']['width'] = style['width']
            feature['properties']['color'] = style['color']

    return roads_data


def set_tags_color(
    tags_data: dict,
    natural_colors: dict,
    landuse_colors: dict,
    color_dict: dict,
) -> tuple:
    """Compute building, natural and landuse styles from in-memory tags data.

    Args:
        tags_data: In-memory GeoJSON dict with tag features.
        natural_colors: Dictionary containing colors for natural features.
        landuse_colors: Dictionary containing colors for landuse features.
        color_dict: Dictionary containing color codes for color mapping.

    Returns:
        Tuple of (building_styles, natural_styles, landuse_styles) dictionaries.
    """
    building_types = set()
    natural_types = set()
    landuse_types = set()

    for feature in tags_data['features']:
        props = feature['properties']
        if 'building' in props and props['building']:
            building_types.add(props['building'])
        if 'natural' in props and props['natural']:
            natural_types.add(props['natural'])
        if 'landuse' in props and props['landuse']:
            landuse_types.add(props['landuse'])

    building_styles = {}
    for building_type in building_types:
        if building_type is not None:
            building_styles[building_type] = {'color': 'brown'}

    natural_styles = {}
    for natural_type in natural_types:
        if natural_type in natural_colors:
            natural_styles[natural_type] = {'color': color_dict[natural_colors[natural_type]]}

    landuse_styles = {}
    for landuse_type in landuse_types:
        if landuse_type in landuse_colors:
            landuse_styles[landuse_type] = {'color': color_dict[landuse_colors[landuse_type]]}

    return building_styles, natural_styles, landuse_styles


def process_tags_data(
    tags_data: dict,
    building_styles: dict,
    natural_styles: dict,
    landuse_styles: dict,
) -> dict:
    """Set colors on tag features in memory.

    Args:
        tags_data: In-memory GeoJSON dict with tag features.
        building_styles: Dictionary containing colors for building features.
        natural_styles: Dictionary containing colors for natural features.
        landuse_styles: Dictionary containing colors for landuse features.

    Returns:
        The mutated tags_data dict.
    """
    osm_class = ["natural", "landuse", "building"]
    osm_class_styles = [natural_styles, landuse_styles, building_styles]

    for feature in tags_data['features']:
        feature['properties']['color'] = '#000000'
        for osm_type, style_dict in zip(osm_class, osm_class_styles):
            if osm_type in feature["properties"]:
                class_type = feature["properties"][osm_type]
                if class_type in style_dict:
                    feature['properties']['color'] = style_dict[class_type]['color']

    return tags_data


def filter_highways_S2(
    roads_data: dict,
    threshold: int,
    s2_path: str,
) -> dict:
    """Filter highways not visible in S2 images using vectorized raster sampling.

    Highways of type "track", "footway", "path", "residential", "living_street"
    are filtered based on their visibility in the S2 B04 band.

    Args:
        roads_data: In-memory GeoJSON dict with highway features.
        threshold: Threshold value to filter features.
        s2_path: Path to the S2 B04 band raster.

    Returns:
        The mutated roads_data dict.
    """
    if not os.path.isfile(s2_path):
        logger.warning("S2 B04 image not found for highway filtering: %s", s2_path)
        return roads_data

    filterable_types = {"track", "footway", "path", "residential", "living_street"}

    with rasterio.open(s2_path) as src:
        s2_array = src.read(1)
        s2_transform = src.transform
        s2_height, s2_width = s2_array.shape
        s2_bounds = src.bounds

        src_crs = pyproj.CRS('EPSG:4326')
        dst_crs = pyproj.CRS(src.crs)
        project = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
        s2_bbox = box(*s2_bounds)

        for feature in roads_data["features"]:
            geom_type = feature.get("geometry", {}).get("type")
            highway_type = feature['properties'].get('highway')

            if geom_type != "LineString" or highway_type not in filterable_types:
                continue

            try:
                line = LineString(feature["geometry"]["coordinates"])
                proj_line = transform(project, line)
                clipped = s2_bbox.intersection(proj_line)

                if clipped.is_empty:
                    continue

                # Sample points along the line at ~10m intervals
                if hasattr(clipped, 'length') and clipped.length > 0:
                    num_samples = max(int(clipped.length / 10), 2)
                    distances = np.linspace(0, clipped.length, num_samples)
                    points = [clipped.interpolate(d) for d in distances]
                    xs = np.array([p.x for p in points])
                    ys = np.array([p.y for p in points])
                else:
                    continue

                rows, cols = rasterio.transform.rowcol(s2_transform, xs, ys)
                rows = np.asarray(rows)
                cols = np.asarray(cols)

                valid = (rows >= 0) & (rows < s2_height) & (cols >= 0) & (cols < s2_width)
                if not np.any(valid):
                    continue

                values = s2_array[rows[valid], cols[valid]]
                nonzero = values[values != 0]

                if nonzero.size > 0 and np.mean(nonzero) < threshold:
                    feature['properties']['color'] = '#000000'

            except Exception as e:
                logger.warning(
                    "Error processing feature %s: %s",
                    feature["properties"].get("osmid", "unknown"), e,
                )

    return roads_data


def combine_geojson_data(tags_data: dict, roads_data: dict) -> dict:
    """Combine tags and roads GeoJSON dicts into a single FeatureCollection.

    Args:
        tags_data: In-memory GeoJSON dict with tag features.
        roads_data: In-memory GeoJSON dict with highway features.

    Returns:
        Combined GeoJSON dict.
    """
    combined_features = tags_data.get('features', []) + roads_data.get('features', [])
    return {'type': 'FeatureCollection', 'features': combined_features}


def _write_geojson(data: dict, output_path: str) -> None:
    """Write a GeoJSON dict to disk.

    Args:
        data: GeoJSON FeatureCollection dict.
        output_path: File path to write.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Utilities (unchanged)
# ---------------------------------------------------------------------------


def clear_and_create_directories(img_folder: str, mask_folder: str) -> None:
    """Clear and create directories.

    Args:
        img_folder: Path to OSM raster directory.
        mask_folder: Path to OSM GeoJSON directory.
    """
    if os.path.exists(img_folder):
        shutil.rmtree(img_folder)
    if os.path.exists(mask_folder):
        shutil.rmtree(mask_folder)

    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)


def color_to_rgb(color_tag: str) -> tuple:
    """Return RGB values based on color tag.

    Args:
        color_tag: The color tag (hex or named).

    Returns:
        Tuple of (R, G, B) values.
    """
    try:
        return ImageColor.getrgb(color_tag)
    except ValueError:
        return (0, 0, 0)


# ---------------------------------------------------------------------------
# rasterize_geojson helpers
# ---------------------------------------------------------------------------

def _add_rgb_class_fields(dataset_path: str) -> None:
    """Open GeoJSON, add RGB/class fields, populate with color values, and save.

    Args:
        dataset_path: Path to GeoJSON file to modify in-place.
    """
    dataset = ogr.Open(dataset_path, 1)
    if not dataset:
        raise FileNotFoundError(f"Cannot open {dataset_path}")

    layer = dataset.GetLayer()

    # Create fields if they don't exist
    layer_defn = layer.GetLayerDefn()
    for field_name in ["R", "G", "B", "class"]:
        if layer_defn.GetFieldIndex(field_name) == -1:
            field_defn = ogr.FieldDefn(field_name, ogr.OFTInteger)
            layer.CreateField(field_defn)

    # Identify unique highway types and create a mapping
    highway_types = set()
    for feature in layer:
        highway_type = feature.GetField("highway")
        if highway_type:
            highway_types.add(highway_type)

    highway_type_mapping = {htype: idx + 2 for idx, htype in enumerate(sorted(highway_types))}

    # Update each feature with RGB values and class
    layer.ResetReading()
    for feature in layer:
        color_tag = feature.GetField("color")
        rgb_value = color_to_rgb(color_tag) if color_tag else (255, 255, 255)

        feature.SetField("R", rgb_value[0])
        feature.SetField("G", rgb_value[1])
        feature.SetField("B", rgb_value[2])

        highway_type = feature.GetField("highway")
        if highway_type in highway_type_mapping:
            feature.SetField("class", highway_type_mapping[highway_type])

        layer.SetFeature(feature)

    dataset = None


def _reproject_layer_if_needed(shape_layer, img_srs):
    """Reproject shape layer to match image CRS if necessary.

    Args:
        shape_layer: OGR layer to reproject.
        img_srs: Target spatial reference system.

    Returns:
        Tuple of (reprojected_layer, mem_ds_or_None). Caller must keep mem_ds alive.
    """
    shape_srs = shape_layer.GetSpatialRef()

    if shape_srs.ExportToWkt() == img_srs.ExportToWkt():
        return shape_layer, None

    logger.info("Reprojecting shapefile to match image CRS")
    # Force traditional (lon, lat) / (easting, northing) axis order. Under
    # GDAL >= 3 an EPSG:4326 SRS defaults to authority order (lat, lon), which
    # would swap the OSM GeoJSON (lon, lat) coordinates during Transform and
    # place every feature at the wrong location.
    shape_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    img_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    crs_transform = osr.CoordinateTransformation(shape_srs, img_srs)
    mem_drv = ogr.GetDriverByName('Memory')
    mem_ds = mem_drv.CreateDataSource('')
    mem_layer = mem_ds.CreateLayer('reprojected', srs=img_srs)

    mem_layer.CreateFields(shape_layer.schema)

    for feature in shape_layer:
        geom = feature.GetGeometryRef()
        geom.Transform(crs_transform)
        new_feature = ogr.Feature(mem_layer.GetLayerDefn())
        new_feature.SetGeometry(geom)
        for i in range(feature.GetFieldCount()):
            new_feature.SetField(feature.GetFieldDefnRef(i).GetNameRef(), feature.GetField(i))
        mem_layer.CreateFeature(new_feature)

    return mem_layer, mem_ds


def _collect_features_by_type(shape_layer) -> list:
    """Separate features into categorized lists by OSM type.

    Returns features in rasterization order: [natural, landuse, highway, building].

    Args:
        shape_layer: OGR layer to categorize.

    Returns:
        List of feature lists in order: [natural, landuse, highway, building].
    """
    building_features = []
    highway_features = []
    landuse_features = []
    natural_features = []

    field_list = [field.name for field in shape_layer.schema]

    shape_layer.ResetReading()
    for feature in shape_layer:
        if "building" in field_list and feature.GetField("building"):
            building_features.append(feature)
        if "highway" in field_list and feature.GetField("highway"):
            highway_features.append(feature)
        if "landuse" in field_list and feature.GetField("landuse"):
            landuse_features.append(feature)
        if "natural" in field_list and feature.GetField("natural"):
            natural_features.append(feature)

    return [natural_features, landuse_features, highway_features, building_features]


def _rasterize_feature_batches(
    feature_lists: list,
    mem_raster,
    img_srs,
    shape_layer,
) -> None:
    """Rasterize categorized feature batches to the in-memory raster.

    Processes features in order: natural, landuse, highway, building (later
    types overwrite earlier ones where they overlap).

    Args:
        feature_lists: List of feature lists from _collect_features_by_type.
        mem_raster: GDAL in-memory raster to write to.
        img_srs: Image spatial reference system.
        shape_layer: Source layer (for schema).
    """
    for feat_list in feature_lists:
        if not feat_list:
            continue

        temp_ds = ogr.GetDriverByName('Memory').CreateDataSource('')
        temp_layer = temp_ds.CreateLayer('temp_layer', srs=img_srs, geom_type=ogr.wkbPolygon)
        temp_layer.CreateFields(shape_layer.schema)

        for feat in feat_list:
            geom = feat.GetGeometryRef()
            red = feat.GetField('R')
            green = feat.GetField('G')
            blue = feat.GetField('B')
            width = feat.GetField('width')
            if isinstance(width, str):
                try:
                    width = float(width.split()[0])
                except (ValueError, IndexError):
                    width = None

            if red is not None and green is not None and blue is not None:
                if geom.GetGeometryType() == ogr.wkbLineString and width is not None:
                    buffer_distance = width / 2
                    buffered_geom = geom.Buffer(buffer_distance)
                else:
                    buffered_geom = geom

                new_feature = ogr.Feature(temp_layer.GetLayerDefn())
                new_feature.SetGeometry(buffered_geom)
                new_feature.SetField('R', red)
                new_feature.SetField('G', green)
                new_feature.SetField('B', blue)
                new_feature.SetField('width', width)
                temp_layer.CreateFeature(new_feature)

        gdal.RasterizeLayer(mem_raster, [1], temp_layer, options=["ATTRIBUTE=R", "ALL_TOUCHED=True"])
        gdal.RasterizeLayer(mem_raster, [2], temp_layer, options=["ATTRIBUTE=G", "ALL_TOUCHED=True"])
        gdal.RasterizeLayer(mem_raster, [3], temp_layer, options=["ATTRIBUTE=B", "ALL_TOUCHED=True"])

        temp_ds = None


def _features_to_memory_ogr_layer(features: list, geom_srs):
    """Build an in-memory OGR layer from styled GeoJSON-like feature dicts.

    Computes R/G/B/class fields directly from each feature's ``color`` and
    ``highway`` properties — the same work ``_add_rgb_class_fields`` does on
    disk, but without an OGR open/iterate/SetField/close cycle on a GeoJSON
    file.

    Args:
        features: List of GeoJSON feature dicts. Features must already be
            styled (i.e. ``properties.color`` set by ``process_highway_data``
            / ``process_tags_data``).
        geom_srs: ``osr.SpatialReference`` for the geometry CRS (typically
            EPSG:4326 for OSM data).

    Returns:
        Tuple ``(mem_ds, mem_layer)``. Caller must keep ``mem_ds`` alive for
        as long as ``mem_layer`` is in use.
    """
    highway_types = sorted({
        (f.get('properties') or {}).get('highway')
        for f in features
        if (f.get('properties') or {}).get('highway')
    })
    highway_class_map = {ht: idx + 2 for idx, ht in enumerate(highway_types)}

    mem_drv = ogr.GetDriverByName('Memory')
    mem_ds = mem_drv.CreateDataSource('osm_features')
    mem_layer = mem_ds.CreateLayer('osm', srs=geom_srs, geom_type=ogr.wkbUnknown)

    field_specs = [
        ('R', ogr.OFTInteger), ('G', ogr.OFTInteger), ('B', ogr.OFTInteger),
        ('class', ogr.OFTInteger), ('width', ogr.OFTReal),
        ('highway', ogr.OFTString), ('building', ogr.OFTString),
        ('landuse', ogr.OFTString), ('natural', ogr.OFTString),
    ]
    for name, ftype in field_specs:
        mem_layer.CreateField(ogr.FieldDefn(name, ftype))

    layer_defn = mem_layer.GetLayerDefn()

    for feat in features:
        geom_dict = feat.get('geometry')
        if not geom_dict:
            continue
        try:
            geom = ogr.CreateGeometryFromJson(json.dumps(geom_dict))
        except Exception:
            continue
        if geom is None:
            continue

        props = feat.get('properties') or {}
        color_tag = props.get('color')
        rgb = color_to_rgb(color_tag) if color_tag else (255, 255, 255)

        new_feature = ogr.Feature(layer_defn)
        new_feature.SetGeometry(geom)
        new_feature.SetField('R', int(rgb[0]))
        new_feature.SetField('G', int(rgb[1]))
        new_feature.SetField('B', int(rgb[2]))

        width = props.get('width')
        if width is not None:
            try:
                new_feature.SetField('width', float(width))
            except (TypeError, ValueError):
                pass

        highway = props.get('highway')
        if highway:
            new_feature.SetField('highway', str(highway))
            cls_idx = highway_class_map.get(highway)
            if cls_idx is not None:
                new_feature.SetField('class', cls_idx)

        for tag in ('building', 'landuse', 'natural'):
            val = props.get(tag)
            if val:
                new_feature.SetField(tag, str(val))

        mem_layer.CreateFeature(new_feature)
        new_feature = None

    return mem_ds, mem_layer


def rasterize_features_in_memory(
    features: list,
    img_path: str,
    output_path: str,
) -> None:
    """Rasterize OSM features directly from in-memory dicts.

    Equivalent to ``rasterize_geojson`` but skips the GeoJSON disk read and
    OGR-update-mode rewrite — the slow part of the legacy path. Features
    must already be styled (``color`` property set).

    Args:
        features: List of GeoJSON feature dicts in EPSG:4326.
        img_path: Reference image whose CRS, extent and resolution define
            the output raster.
        output_path: Path to output OSM raster (GTiff).
    """
    validate_file_exists(img_path, "image file")

    factor = RASTERIZE_UPSAMPLE_FACTOR
    gdal.AllRegister()

    img_ds = gdal.Open(img_path, gdal.GA_ReadOnly)
    img_srs = osr.SpatialReference(wkt=img_ds.GetProjection())

    src_srs = osr.SpatialReference()
    src_srs.ImportFromEPSG(4326)

    src_ds, src_layer = _features_to_memory_ogr_layer(features, src_srs)

    shape_layer, reproject_ds = _reproject_layer_if_needed(src_layer, img_srs)

    ulx, xres, xskew, uly, yskew, yres = img_ds.GetGeoTransform()
    new_geo_transform = (ulx, xres / factor, xskew, uly, yskew, yres / factor)
    mem_drv = gdal.GetDriverByName('MEM')
    mem_raster = mem_drv.Create(
        '', img_ds.RasterXSize * factor, img_ds.RasterYSize * factor,
        3, gdal.GDT_Byte,
    )
    mem_raster.SetProjection(img_ds.GetProjection())
    mem_raster.SetGeoTransform(new_geo_transform)

    for i in range(1, 4):
        mem_band = mem_raster.GetRasterBand(i)
        mem_band.Fill(0)
        mem_band.SetNoDataValue(0)

    feature_lists = _collect_features_by_type(shape_layer)
    _rasterize_feature_batches(feature_lists, mem_raster, img_srs, shape_layer)

    roi = mem_raster.ReadAsArray()
    roi = roi.transpose(1, 2, 0)

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, roi.shape[1], roi.shape[0], 3, gdal.GDT_Byte)
    out_ds.SetGeoTransform(mem_raster.GetGeoTransform())
    out_ds.SetProjection(mem_raster.GetProjection())

    for i in range(3):
        out_ds.GetRasterBand(i + 1).WriteArray(roi[:, :, i])

    out_ds.FlushCache()
    out_ds = None


def rasterize_geojson(geojson_path: str, img_path: str, output_path: str) -> None:
    """Rasterize the combined GeoJSON file.

    Args:
        geojson_path: Path to input OSM GeoJSON file.
        img_path: Path to corresponding image file.
        output_path: Path to output OSM raster.
    """
    validate_file_exists(geojson_path, "GeoJSON file")
    validate_file_exists(img_path, "image file")

    factor = RASTERIZE_UPSAMPLE_FACTOR
    gdal.AllRegister()

    # 1. Add RGB and class fields to GeoJSON features
    _add_rgb_class_fields(geojson_path)

    # 2. Open image and shape datasets
    img_ds = gdal.Open(img_path, gdal.GA_ReadOnly)
    shape_dataset = ogr.Open(geojson_path)
    shape_layer = shape_dataset.GetLayer()

    img_srs = osr.SpatialReference(wkt=img_ds.GetProjection())

    # 3. Reproject if necessary
    shape_layer, reproject_ds = _reproject_layer_if_needed(shape_layer, img_srs)

    # 4. Create in-memory raster with upsampled resolution
    ulx, xres, xskew, uly, yskew, yres = img_ds.GetGeoTransform()
    new_geo_transform = (ulx, xres / factor, xskew, uly, yskew, yres / factor)
    mem_drv = gdal.GetDriverByName('MEM')
    mem_raster = mem_drv.Create('', img_ds.RasterXSize * factor, img_ds.RasterYSize * factor, 3, gdal.GDT_Byte)
    mem_raster.SetProjection(img_ds.GetProjection())
    mem_raster.SetGeoTransform(new_geo_transform)

    for i in range(1, 4):
        mem_band = mem_raster.GetRasterBand(i)
        mem_band.Fill(0)
        mem_band.SetNoDataValue(0)

    # 5. Collect and rasterize features by type
    feature_lists = _collect_features_by_type(shape_layer)
    _rasterize_feature_batches(feature_lists, mem_raster, img_srs, shape_layer)

    # 6. Save output
    roi = mem_raster.ReadAsArray()
    roi = roi.transpose(1, 2, 0)

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, roi.shape[1], roi.shape[0], 3, gdal.GDT_Byte)
    out_ds.SetGeoTransform(mem_raster.GetGeoTransform())
    out_ds.SetProjection(mem_raster.GetProjection())

    for i in range(3):
        out_ds.GetRasterBand(i + 1).WriteArray(roi[:, :, i])

    out_ds.FlushCache()
    out_ds = None


# ---------------------------------------------------------------------------
# Public orchestrators (signatures unchanged for callers)
# ---------------------------------------------------------------------------


def create_osm_raster(
    image_path: str,
    output_directory: str,
    highway_colors: dict,
    natural_colors: dict,
    landuse_colors: dict,
    color_dict: dict,
    highway_threshold: int,
) -> None:
    """Full processing to extract and rasterize OSM data from image bounds.

    Args:
        image_path: Path to input image.
        output_directory: Path to output directory.
        highway_colors: Dictionary containing colors for highway features.
        natural_colors: Dictionary containing colors for natural features.
        landuse_colors: Dictionary containing colors for landuse features.
        color_dict: Dictionary containing color codes for color mapping.
        highway_threshold: Threshold value to remove invisible highway features.
    """
    validate_file_exists(image_path, "input image")

    image_name = get_stem(image_path)

    # 1. Single Overpass API call — returns in-memory GeoJSON dicts
    bbox = extract_bbox_from_image(image_path)
    query_tags = build_osm_query_tags(highway_colors, natural_colors, landuse_colors)
    tags_data, roads_data = extract_osm_data(bbox, image_name, query_tags)

    # 2. Compute and apply styles in memory
    highway_styles = set_highway_styles(roads_data, highway_colors, color_dict, DEFAULT_HIGHWAY_WIDTH)
    roads_data = process_highway_data(roads_data, highway_styles)

    building_styles, natural_styles, landuse_styles = set_tags_color(
        tags_data, natural_colors, landuse_colors, color_dict,
    )
    tags_data = process_tags_data(tags_data, building_styles, natural_styles, landuse_styles)

    # 3. Filter invisible highways using vectorized S2 sampling
    s2_b04_path = os.path.abspath(
        os.path.join(output_directory, "..", "B04", "S2_crop_extended", f"{image_name}.tif")
    )
    try:
        roads_data = filter_highways_S2(roads_data, highway_threshold, s2_b04_path)
    except Exception as e:
        logger.warning("Highway filter failed for %s: %s", image_name, e)

    # 4. Combine and write single GeoJSON to disk (required by OGR rasterizer)
    combined_data = combine_geojson_data(tags_data, roads_data)
    combined_geojson_path = os.path.join(
        output_directory, "geojson", f"{image_name}_complete.geojson",
    )
    _write_geojson(combined_data, combined_geojson_path)

    # 5. Rasterize directly from in-memory features (avoids GeoJSON re-open)
    all_mask_output_path = os.path.join(output_directory, "image", f"{image_name}.tif")
    rasterize_features_in_memory(combined_data['features'], image_path, all_mask_output_path)


def create_osm_raster_from_bounds(
    chip_name: str,
    bbox_4326: tuple,
    ref_raster_path: str,
    output_directory: str,
    highway_colors: dict,
    natural_colors: dict,
    landuse_colors: dict,
    color_dict: dict,
) -> None:
    """Extract and rasterize OSM data for a chip defined by its bounds.

    Similar to ``create_osm_raster`` but works from a pre-computed bounding
    box and a reference raster instead of an S2 image.  Highway S2-based
    filtering is skipped (no satellite image available).

    Args:
        chip_name: Identifier used to name intermediate and output files.
        bbox_4326: Bounding box in EPSG:4326, canonical (west, south, east, north).
        ref_raster_path: Path to a blank reference raster that defines the
            CRS, extent and resolution for rasterization.
        output_directory: Root output directory for this product.
        highway_colors: Highway color mapping dict.
        natural_colors: Natural feature color mapping dict.
        landuse_colors: Landuse color mapping dict.
        color_dict: Master color-name-to-hex dict.
    """
    validate_file_exists(ref_raster_path, "reference raster")

    # 1. Single Overpass API call
    query_tags = build_osm_query_tags(highway_colors, natural_colors, landuse_colors)
    tags_data, roads_data = extract_osm_data(bbox_4326, chip_name, query_tags)

    # 2. Compute and apply styles in memory
    highway_styles = set_highway_styles(roads_data, highway_colors, color_dict, DEFAULT_HIGHWAY_WIDTH)
    roads_data = process_highway_data(roads_data, highway_styles)

    building_styles, natural_styles, landuse_styles = set_tags_color(
        tags_data, natural_colors, landuse_colors, color_dict,
    )
    tags_data = process_tags_data(tags_data, building_styles, natural_styles, landuse_styles)

    # 3. No S2-based highway filtering in AOI mode

    # 4. Combine and write single GeoJSON to disk
    combined_data = combine_geojson_data(tags_data, roads_data)
    combined_geojson_path = os.path.join(
        output_directory, "geojson", f"{chip_name}_complete.geojson",
    )
    _write_geojson(combined_data, combined_geojson_path)

    # 5. Rasterize directly from in-memory features (avoids GeoJSON re-open)
    output_raster_path = os.path.join(output_directory, "image", f"{chip_name}.tif")
    rasterize_features_in_memory(combined_data['features'], ref_raster_path, output_raster_path)


# ---------------------------------------------------------------------------
# Batched orchestrators: one Overpass call covering many chips
# ---------------------------------------------------------------------------


def _cluster_chips_by_grid(centers: list, max_deg: float) -> list:
    """Group chips into spatial clusters whose union bbox is bounded.

    Each cluster is a grid cell of size ``max_deg`` × ``max_deg``, so the
    Overpass call for that cluster covers at most ``max_deg`` per side and
    the resulting GeoDataFrame stays bounded in memory.

    Args:
        centers: List of ``(lon, lat)`` tuples for chip centers.
        max_deg: Maximum cluster cell size in degrees. ``<= 0`` disables
            clustering and returns a single cluster with all chips.

    Returns:
        List of clusters, each a list of indices into ``centers``.
    """
    if max_deg <= 0 or not centers:
        return [list(range(len(centers)))] if centers else []
    cells: dict = {}
    for i, (lon, lat) in enumerate(centers):
        cell = (int(lon // max_deg), int(lat // max_deg))
        cells.setdefault(cell, []).append(i)
    return list(cells.values())


def _process_chip_from_shared_gdf(
    gdf_extent: "gpd.GeoDataFrame",
    chip_name: str,
    chip_box_4326,
    raster_ref_path: str,
    output_directory: str,
    highway_colors: dict,
    natural_colors: dict,
    landuse_colors: dict,
    color_dict: dict,
    s2_b04_path: Optional[str],
    highway_threshold: Optional[int],
) -> None:
    """Subset, style, optionally filter, write GeoJSON and rasterize one chip.

    Shared per-chip body for the two batched orchestrators.

    Args:
        gdf_extent: Shared GeoDataFrame returned by
            ``extract_osm_data_for_extent``.
        chip_name: Identifier used to name intermediate and output files.
        chip_box_4326: Shapely geometry (typically a box) in EPSG:4326
            defining the chip footprint for spatial subset.
        raster_ref_path: Reference raster (S2 chip image or AOI ref raster)
            used by the rasterizer to set CRS/extent/resolution.
        output_directory: OSM output directory containing ``geojson/`` and
            ``image/`` subfolders.
        highway_colors / natural_colors / landuse_colors / color_dict:
            Style dictionaries (passed through unchanged).
        s2_b04_path: Path to the chip's S2 B04 raster, used for the
            highway-visibility filter. Pass ``None`` to skip the filter
            (AOI mode).
        highway_threshold: Threshold for the highway-visibility filter.
            Ignored when ``s2_b04_path`` is ``None``.
    """
    tags_data, roads_data = _gdf_subset_to_geojson_dicts(gdf_extent, chip_box_4326)

    highway_styles = set_highway_styles(
        roads_data, highway_colors, color_dict, DEFAULT_HIGHWAY_WIDTH,
    )
    roads_data = process_highway_data(roads_data, highway_styles)

    building_styles, natural_styles, landuse_styles = set_tags_color(
        tags_data, natural_colors, landuse_colors, color_dict,
    )
    tags_data = process_tags_data(
        tags_data, building_styles, natural_styles, landuse_styles,
    )

    if s2_b04_path is not None and highway_threshold is not None:
        try:
            roads_data = filter_highways_S2(roads_data, highway_threshold, s2_b04_path)
        except Exception as e:
            logger.warning("Highway filter failed for %s: %s", chip_name, e)

    combined_data = combine_geojson_data(tags_data, roads_data)
    combined_geojson_path = os.path.join(
        output_directory, "geojson", f"{chip_name}_complete.geojson",
    )
    _write_geojson(combined_data, combined_geojson_path)

    output_raster_path = os.path.join(
        output_directory, "image", f"{chip_name}.tif",
    )
    rasterize_features_in_memory(
        combined_data['features'], raster_ref_path, output_raster_path,
    )


def create_osm_rasters_for_chips(
    chip_paths: list,
    output_directory: str,
    highway_colors: dict,
    natural_colors: dict,
    landuse_colors: dict,
    color_dict: dict,
    highway_threshold: int,
    batch_max_deg: float = 0.2,
) -> None:
    """Batched OSM rasterization for S2/full-tile chip images.

    Groups chips into spatial clusters of at most ``batch_max_deg`` per
    side and issues one Overpass call per cluster. Replaces N per-chip
    Overpass calls with ~1 per cluster, while keeping the per-call
    GeoDataFrame bounded in memory (a single union over a full S2 tile
    can be multiple GB).

    On Overpass failure for a cluster, falls back to per-chip
    ``create_osm_raster`` for that cluster only.

    Args:
        chip_paths: List of S2 chip image paths.
        output_directory: OSM output directory (must contain ``geojson/``
            and ``image/`` subfolders).
        highway_colors / natural_colors / landuse_colors / color_dict:
            Style dictionaries (passed through unchanged).
        highway_threshold: Threshold for the highway-visibility filter.
        batch_max_deg: Maximum bbox extent (in degrees) per Overpass
            batch. Lower values cap memory at the cost of more HTTP calls.
            ``<= 0`` disables clustering (single batch over all chips).
    """
    if not chip_paths:
        return

    # Per-chip bboxes in EPSG:4326
    chip_bboxes = []
    for path in chip_paths:
        try:
            bbox = extract_bbox_from_image(path)
            chip_bboxes.append((path, bbox))
        except Exception:
            logger.exception("Failed to extract bbox from %s", path)

    if not chip_bboxes:
        return

    # bbox tuple from extract_bbox_from_image is canonical (w, s, e, n)
    centers = [
        ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for _, b in chip_bboxes
    ]
    clusters = _cluster_chips_by_grid(centers, batch_max_deg)
    query_tags = build_osm_query_tags(highway_colors, natural_colors, landuse_colors)
    logger.info(
        "Processing %d chips in %d Overpass batch(es) (max %.3f deg per batch)",
        len(chip_bboxes), len(clusters), batch_max_deg,
    )

    for cluster_idx, indices in enumerate(clusters, start=1):
        cluster_bboxes = [chip_bboxes[i] for i in indices]
        wests = [b[0] for _, b in cluster_bboxes]
        souths = [b[1] for _, b in cluster_bboxes]
        easts = [b[2] for _, b in cluster_bboxes]
        norths = [b[3] for _, b in cluster_bboxes]
        union_bbox = (min(wests), min(souths), max(easts), max(norths))
        extent_name = f"batch {cluster_idx}/{len(clusters)} ({len(cluster_bboxes)} chips)"
        logger.info(
            "Fetching OSM data for %s: bbox=%s", extent_name, union_bbox,
        )

        gdf_extent = None
        try:
            gdf_extent = extract_osm_data_for_extent(union_bbox, extent_name, query_tags)
        except (requests.exceptions.RequestException, ResponseStatusCodeError) as e:
            # Network / HTTP failure (rate limit, 502 "server at capacity", ...),
            # already retried with backoff by download_features: do NOT fall back
            # to per-chip, which would multiply requests against an endpoint
            # already refusing us. Stop cleanly so a later rerun resumes the
            # remaining chips.
            raise RuntimeError(
                f"Overpass request failed for {extent_name} "
                f"({e.__class__.__name__}). The endpoint is overloaded or "
                "rate-limited this IP. Wait and rerun to resume the remaining chips, "
                "or set GENCP_OVERPASS_URL to a mirror / local Overpass instance "
                "(comma-separated list to rotate)."
            ) from e
        except Exception:
            logger.exception(
                "Batched OSM fetch failed for %s — falling back to per-chip", extent_name,
            )

        if gdf_extent is None:
            for path, _ in cluster_bboxes:
                try:
                    create_osm_raster(
                        path, output_directory, highway_colors, natural_colors,
                        landuse_colors, color_dict, highway_threshold,
                    )
                except Exception:
                    logger.exception("Per-chip fallback failed for %s", path)
            continue

        for path, bbox in cluster_bboxes:
            try:
                chip_name = get_stem(path)
                w, s, e, n = bbox
                chip_box = box(min(w, e), min(s, n), max(w, e), max(s, n))
                s2_b04_path = os.path.abspath(
                    os.path.join(
                        output_directory, "..", "B04", "S2_crop_extended", f"{chip_name}.tif",
                    )
                )
                _process_chip_from_shared_gdf(
                    gdf_extent, chip_name, chip_box, path, output_directory,
                    highway_colors, natural_colors, landuse_colors, color_dict,
                    s2_b04_path, highway_threshold,
                )
            except Exception:
                logger.exception("Batched OSM rasterization failed for chip: %s", path)

        # Drop the GDF before fetching the next batch so peak memory is bounded
        # by a single cluster, not the whole pipeline.
        del gdf_extent


def create_osm_rasters_for_chips_from_bounds(
    chips: list,
    ref_paths: dict,
    output_directory: str,
    highway_colors: dict,
    natural_colors: dict,
    landuse_colors: dict,
    color_dict: dict,
    batch_max_deg: float = 0.2,
) -> None:
    """Batched OSM rasterization for AOI-mode chips.

    Equivalent of ``create_osm_rasters_for_chips`` for the AOI pipeline.
    Chips are grouped into spatial clusters bounded by ``batch_max_deg``
    per side, with one Overpass call per cluster. No S2-based highway
    filter (no satellite image available in AOI mode).

    On Overpass failure for a cluster, falls back to per-chip
    ``create_osm_raster_from_bounds`` for that cluster only.

    Args:
        chips: List of dicts with at least ``name`` and ``bbox_4326``
            entries. ``bbox_4326`` follows the canonical convention
            ``(west, south, east, north)``.
        ref_paths: Mapping from chip name to its reference raster path.
        output_directory: OSM output directory containing ``geojson/`` and
            ``image/`` subfolders.
        highway_colors / natural_colors / landuse_colors / color_dict:
            Style dictionaries (passed through unchanged).
        batch_max_deg: Maximum bbox extent (in degrees) per Overpass
            batch. ``<= 0`` disables clustering.
    """
    if not chips:
        return

    # AOI bbox tuple: canonical (west, south, east, north)
    centers = []
    for c in chips:
        w, s, e, n = c['bbox_4326']
        centers.append(((w + e) / 2, (n + s) / 2))
    clusters = _cluster_chips_by_grid(centers, batch_max_deg)
    query_tags = build_osm_query_tags(highway_colors, natural_colors, landuse_colors)
    logger.info(
        "Processing %d AOI chips in %d Overpass batch(es) (max %.3f deg per batch)",
        len(chips), len(clusters), batch_max_deg,
    )

    for cluster_idx, indices in enumerate(clusters, start=1):
        cluster_chips = [chips[i] for i in indices]
        wests = [c['bbox_4326'][0] for c in cluster_chips]
        souths = [c['bbox_4326'][1] for c in cluster_chips]
        easts = [c['bbox_4326'][2] for c in cluster_chips]
        norths = [c['bbox_4326'][3] for c in cluster_chips]
        union_bbox = (min(wests), min(souths), max(easts), max(norths))
        extent_name = (
            f"AOI batch {cluster_idx}/{len(clusters)} ({len(cluster_chips)} chips)"
        )
        logger.info("Fetching OSM data for %s: bbox=%s", extent_name, union_bbox)

        gdf_extent = None
        try:
            gdf_extent = extract_osm_data_for_extent(union_bbox, extent_name, query_tags)
        except (requests.exceptions.RequestException, ResponseStatusCodeError) as e:
            # Network / HTTP failure (rate limit, 502 "server at capacity", ...),
            # already retried with backoff by download_features: do NOT fall back
            # to per-chip, which would multiply requests against an endpoint
            # already refusing us. Stop cleanly so a later rerun resumes the
            # remaining chips.
            raise RuntimeError(
                f"Overpass request failed for {extent_name} "
                f"({e.__class__.__name__}). The endpoint is overloaded or "
                "rate-limited this IP. Wait and rerun to resume the remaining chips, "
                "or set GENCP_OVERPASS_URL to a mirror / local Overpass instance "
                "(comma-separated list to rotate)."
            ) from e
        except Exception:
            logger.exception(
                "Batched OSM fetch failed for %s — falling back to per-chip", extent_name,
            )

        if gdf_extent is None:
            for chip in cluster_chips:
                ref_path = ref_paths.get(chip['name'])
                if ref_path is None:
                    continue
                try:
                    create_osm_raster_from_bounds(
                        chip['name'], chip['bbox_4326'], ref_path, output_directory,
                        highway_colors, natural_colors, landuse_colors, color_dict,
                    )
                except Exception:
                    logger.exception("Per-chip AOI fallback failed for %s", chip['name'])
            continue

        for chip in cluster_chips:
            chip_name = chip['name']
            ref_path = ref_paths.get(chip_name)
            if ref_path is None:
                continue
            try:
                w, s, e, n = chip['bbox_4326']
                chip_box = box(min(w, e), min(s, n), max(w, e), max(s, n))
                _process_chip_from_shared_gdf(
                    gdf_extent, chip_name, chip_box, ref_path, output_directory,
                    highway_colors, natural_colors, landuse_colors, color_dict,
                    None, None,
                )
            except Exception:
                logger.exception("Batched OSM rasterization failed for chip: %s", chip_name)

        del gdf_extent
