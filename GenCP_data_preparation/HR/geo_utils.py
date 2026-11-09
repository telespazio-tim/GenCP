"""Shared path manipulation, validation, and geospatial utilities."""

import logging
import re
from pathlib import Path
from typing import List, Tuple, Union

import geopandas as gpd
import pyproj
import rasterio
import rasterio.mask
import rasterio.warp
from osgeo import gdal, osr
from rasterio.transform import from_bounds
from shapely.geometry import box

from ..common.geo import bbox_from_vector, estimate_utm_crs

logger = logging.getLogger(__name__)


def get_stem(path: Union[str, Path]) -> str:
    """Get filename without extension.

    Replaces fragile patterns like ``os.path.basename(x)[:-4]``.
    """
    return Path(path).stem


def extract_mgrs_tile(product_name: str) -> str:
    """Extract the MGRS tile code from an S2 product name.

    S2 product names follow:
        S2A_MSIL2A_YYYYMMDDTHHMMSS_NXXXX_RXXX_TXXXXX_...
    The MGRS tile is in the ``T``-prefixed field (e.g. ``T31TEJ``).

    Replaces the fragile ``os.path.basename(S2_path)[39:45]`` pattern.
    """
    match = re.search(r"_T(\d{2}[A-Z]{3})_", product_name)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot extract MGRS tile from product name: {product_name}")


def extract_band_basename(band_path: str) -> str:
    """Extract band filename prefix by removing the ``_Bxx_10m.TIF`` suffix.

    Replaces ``os.path.basename(red)[:-12]`` in quicklook.
    """
    name = Path(band_path).name
    match = re.match(r"(.+)_B\d{2}_10m\.TIF$", name)
    if match:
        return match.group(1)
    return name[:-12]


def validate_file_exists(path: Union[str, Path], description: str = "file") -> Path:
    """Validate that a file exists, raising FileNotFoundError with a clear message."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Required {description} not found: {p}")
    return p


def validate_dir_exists(path: Union[str, Path], description: str = "directory") -> Path:
    """Validate that a directory exists, raising FileNotFoundError with a clear message."""
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"Required {description} not found: {p}")
    return p


def extract_bbox_from_vector(vector_path: Union[str, Path]) -> Tuple[tuple, str]:
    """Read a GeoJSON or shapefile and return its total bounds in EPSG:4326.

    Delegates to :func:`GenCP_data_preparation.common.geo.bbox_from_vector`.

    Args:
        vector_path: Path to a GeoJSON or shapefile.

    Returns:
        Tuple of (bbox_4326, native_crs_epsg).
        bbox_4326 is canonical ``(west, south, east, north)`` (as osmnx 2.x
        expects). native_crs_epsg is the EPSG string of the vector's CRS.
    """
    bbox_4326, native_crs = bbox_from_vector(vector_path)
    return bbox_4326, str(native_crs)


def _estimate_utm_crs(lon: float, lat: float) -> pyproj.CRS:
    """Estimate an appropriate UTM CRS for a given lon/lat point.

    Thin wrapper kept for internal callers; delegates to the shared
    :func:`GenCP_data_preparation.common.geo.estimate_utm_crs`.
    """
    return estimate_utm_crs(lon, lat)


def generate_chip_grid(
    vector_path: Union[str, Path],
    chip_size_meters: float,
    resolution_meters: float = 10.0,
) -> List[dict]:
    """Generate a regular grid of chips covering the AOI polygon(s).

    Each chip is a square of ``chip_size_meters`` in a projected CRS.
    Only chips that intersect the AOI geometry are returned.

    Args:
        vector_path: Path to a GeoJSON or shapefile defining the AOI.
        chip_size_meters: Side length of each chip in meters.
        resolution_meters: Pixel size in meters (used to compute raster dimensions).

    Returns:
        List of dicts, each containing:
        - ``name``: chip identifier (e.g. "chip_0003_0012")
        - ``bounds``: (left, bottom, right, top) in the projected CRS
        - ``crs``: pyproj.CRS of the projected coordinate system
        - ``width``: raster width in pixels
        - ``height``: raster height in pixels
        - ``transform``: rasterio-compatible Affine transform
    """
    validate_file_exists(vector_path, "AOI vector file")
    gdf = gpd.read_file(vector_path)
    if gdf.empty:
        raise ValueError(f"AOI vector file is empty: {vector_path}")

    # Project to a metric CRS for regular gridding
    centroid_4326 = gdf.to_crs("EPSG:4326").dissolve().centroid.iloc[0]
    utm_crs = _estimate_utm_crs(centroid_4326.x, centroid_4326.y)
    gdf_proj = gdf.to_crs(utm_crs)
    aoi_union = gdf_proj.union_all()

    minx, miny, maxx, maxy = aoi_union.bounds
    pixel_count = int(chip_size_meters / resolution_meters)

    chips = []
    col_idx = 0
    x = minx
    while x < maxx:
        row_idx = 0
        y = miny
        while y < maxy:
            chip_box = box(x, y, x + chip_size_meters, y + chip_size_meters)
            if aoi_union.intersects(chip_box):
                chip_bounds = (x, y, x + chip_size_meters, y + chip_size_meters)
                chips.append({
                    "name": f"chip_{col_idx:04d}_{row_idx:04d}",
                    "bounds": chip_bounds,
                    "crs": utm_crs,
                    "width": pixel_count,
                    "height": pixel_count,
                    "transform": from_bounds(*chip_bounds, pixel_count, pixel_count),
                })
            row_idx += 1
            y += chip_size_meters
        col_idx += 1
        x += chip_size_meters

    logger.info(
        "Generated %d chips (%.0fm grid) covering AOI from %s",
        len(chips), chip_size_meters, vector_path,
    )
    return chips


def generate_chip_grid_from_bounds(
    bounds: Tuple[float, float, float, float],
    crs: pyproj.CRS,
    chip_size_meters: float,
    resolution_meters: float = 10.0,
    name_prefix: str = "tile",
) -> List[dict]:
    """Generate a regular grid of chips covering a bounding box in a projected CRS.

    Like ``generate_chip_grid`` but operates directly on bounds + CRS rather
    than reading a vector file, and returns every chip in the bounding box
    (no AOI intersection filter).

    Args:
        bounds: (left, bottom, right, top) in the projected CRS units (meters).
        crs: Projected CRS the bounds are expressed in.
        chip_size_meters: Side length of each chip in meters.
        resolution_meters: Pixel size in meters.
        name_prefix: Prefix used in chip names (e.g. "tile" → "tile_0003_0012").

    Returns:
        List of chip dicts (same shape as ``generate_chip_grid``).
    """
    minx, miny, maxx, maxy = bounds
    pixel_count = int(chip_size_meters / resolution_meters)

    chips = []
    col_idx = 0
    x = minx
    while x < maxx:
        row_idx = 0
        y = miny
        while y < maxy:
            chip_bounds = (x, y, x + chip_size_meters, y + chip_size_meters)
            chips.append({
                "name": f"{name_prefix}_{col_idx:04d}_{row_idx:04d}",
                "bounds": chip_bounds,
                "crs": crs,
                "width": pixel_count,
                "height": pixel_count,
                "transform": from_bounds(*chip_bounds, pixel_count, pixel_count),
            })
            row_idx += 1
            y += chip_size_meters
        col_idx += 1
        x += chip_size_meters

    logger.info(
        "Generated %d chips (%.0fm grid) covering bounds %s",
        len(chips), chip_size_meters, bounds,
    )
    return chips


def create_reference_raster(chip: dict, output_path: str) -> str:
    """Create a blank single-band GeoTIFF acting as a spatial reference.

    This raster is used by ``rasterize_geojson`` to define the CRS,
    extent and resolution when no S2 image is available.

    Args:
        chip: A chip dict as returned by ``generate_chip_grid``.
        output_path: Where to write the reference raster.

    Returns:
        The output_path.
    """
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(chip["crs"].to_epsg())

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(output_path, chip["width"], chip["height"], 3, gdal.GDT_Byte)
    ds.SetProjection(srs.ExportToWkt())

    left, bottom, right, top = chip["bounds"]
    pixel_size_x = (right - left) / chip["width"]
    pixel_size_y = (top - bottom) / chip["height"]
    ds.SetGeoTransform((left, pixel_size_x, 0, top, 0, -pixel_size_y))

    for i in range(1, 4):
        band = ds.GetRasterBand(i)
        band.Fill(0)
        band.SetNoDataValue(0)

    ds.FlushCache()
    ds = None
    return output_path


def crop_raster_to_bounds(
    src_raster_path: str,
    bounds,
    output_path: str,
    src_crs=None,
    buffer_distance: float = 0,
) -> str:
    """Crop a raster to given bounds and save.

    Handles optional CRS reprojection of bounds and buffering.
    Common pattern shared by CLC, topography, and S2 crop extraction.

    Args:
        src_raster_path: Path to source raster.
        bounds: Bounding box as (left, bottom, right, top).
        output_path: Path for output cropped raster.
        src_crs: CRS of the input bounds (if different from raster CRS,
                 bounds are reprojected).
        buffer_distance: Optional buffer in raster CRS units to add around bounds.

    Returns:
        Path to the output raster.
    """
    with rasterio.open(src_raster_path) as src:
        target_bounds = bounds
        if src_crs and str(src_crs) != str(src.crs):
            target_bounds = rasterio.warp.transform_bounds(
                src_crs, src.crs, *bounds
            )

        mask_geom = box(*target_bounds)
        if buffer_distance > 0:
            mask_geom = mask_geom.buffer(
                buffer_distance, cap_style=3, join_style=2
            )

        image_crop, transformed = rasterio.mask.mask(src, [mask_geom], crop=True)
        out_profile = src.profile.copy()

    out_profile.update(
        {
            "width": image_crop.shape[2],
            "height": image_crop.shape[1],
            "transform": transformed,
        }
    )
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(image_crop)

    return output_path
