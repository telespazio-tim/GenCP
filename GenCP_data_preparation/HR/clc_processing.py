"""CLC (Copernicus Land Cover) gap-filling for OSM rasters."""

import logging
import os

import numpy as np
import rasterio
import rasterio.mask
import rasterio.warp
from osgeo import gdal
from PIL import ImageColor
from shapely.geometry import box

from .config import RASTERIZE_UPSAMPLE_FACTOR, TARGET_RESOLUTION
from .geo_utils import get_stem, validate_file_exists

logger = logging.getLogger(__name__)


def extract_CLC_crop(
    osm_raster_path: str,
    CLC_path: str,
    tmp_dir: str,
    output_dir: str,
) -> str:
    """Extract CLC raster matching OSM raster bounds.

    Args:
        osm_raster_path: Path to input OSM raster.
        CLC_path: Path to full CLC raster.
        tmp_dir: Path to temporary directory.
        output_dir: Path to output directory.

    Returns:
        Path to output CLC raster.
    """
    validate_file_exists(osm_raster_path, "input OSM raster")
    validate_file_exists(CLC_path, "CLC raster")

    osm_filename = get_stem(osm_raster_path)
    with rasterio.open(osm_raster_path) as src:
        bounds = src.bounds
        src_crs = src.crs

    # Extract CLC crop from OSM raster bounds
    with rasterio.open(CLC_path) as src_clc:
        crs_clc = src_clc.crs
        bounds_clc = rasterio.warp.transform_bounds(src_crs, crs_clc, *bounds)
        mask = [box(*bounds_clc).buffer(10, cap_style=3, join_style=2)]

        image_crop, transformed = rasterio.mask.mask(src_clc, mask, crop=True)
        out_profile = src_clc.profile.copy()

    # Save CLC crop in original CRS
    tmp_clc_crop = os.path.join(tmp_dir, f"{osm_filename}_CLC_crop_tmp.tif")
    out_profile.update({
        "width": image_crop.shape[2],
        "height": image_crop.shape[1],
        "transform": transformed,
    })
    with rasterio.open(tmp_clc_crop, "w", **out_profile) as dst:
        dst.write(image_crop)

    # Convert CLC crop to OSM raster CRS
    tmp_projected_clc_crop = os.path.join(
        tmp_dir, f"{osm_filename}_CLC_crop_projected_tmp.tif"
    )
    input_raster = gdal.Open(tmp_clc_crop)
    warp = gdal.Warp(tmp_projected_clc_crop, input_raster, dstSRS=src_crs)
    warp = None  # Closes the files

    # Crop again projected CLC raster to match OSM raster bounds in same CRS
    with rasterio.open(tmp_projected_clc_crop) as src_clc:
        mask = [box(*bounds)]
        image_crop, transformed = rasterio.mask.mask(src_clc, mask, crop=True)
        out_profile = src_clc.profile.copy()

    # Save CLC crop in OSM raster CRS
    out_image = os.path.join(output_dir, f"{osm_filename}_CLC_crop.tif")
    out_profile.update({
        "width": image_crop.shape[2],
        "height": image_crop.shape[1],
        "transform": transformed,
    })
    with rasterio.open(out_image, "w", **out_profile) as dst:
        dst.write(image_crop)

    return out_image


def fill_osm_no_data(
    osm_raster_path: str,
    clc_crop_path: str,
    output_dir: str,
    CLC_color_mapping: dict,
    color_dict: dict,
) -> None:
    """Fill missing OSM data with CLC data.

    Uses a vectorized numpy lookup table for efficient gap-filling.

    Args:
        osm_raster_path: Path to input OSM raster.
        clc_crop_path: Path to crop CLC raster.
        output_dir: Path to output directory.
        CLC_color_mapping: Dictionary containing colors for CLC values.
        color_dict: Dictionary containing color codes for color mapping.
    """
    validate_file_exists(osm_raster_path, "input OSM raster")
    validate_file_exists(clc_crop_path, "CLC crop raster")

    osm_filename = get_stem(osm_raster_path)
    with rasterio.open(clc_crop_path) as src_lc:
        clc_band = src_lc.read()

    with rasterio.open(osm_raster_path) as src_osm:
        osm_raster = src_osm.read()

    # Build color lookup table: CLC int value -> (R, G, B)
    color_lut = np.zeros((256, 3), dtype=np.uint8)
    for clc_str, color_name in CLC_color_mapping.items():
        try:
            rgb = ImageColor.getrgb(color_dict[color_name])
            color_lut[int(clc_str)] = rgb
        except (KeyError, ValueError) as e:
            logger.warning("Unknown CLC mapping for value %s: %s", clc_str, e)

    # Find all black (no-data) pixels
    nodata_mask = (osm_raster[0] == 0) & (osm_raster[1] == 0) & (osm_raster[2] == 0)
    nodata_i, nodata_j = np.where(nodata_mask)

    # Vectorized fill: downsample indices and batch-lookup colors
    downsampled_i = nodata_i // RASTERIZE_UPSAMPLE_FACTOR
    downsampled_j = nodata_j // RASTERIZE_UPSAMPLE_FACTOR
    clc_vals = clc_band[0, downsampled_i, downsampled_j]
    colors = color_lut[clc_vals]  # (N, 3) array
    for band_idx in range(3):
        osm_raster[band_idx, nodata_i, nodata_j] = colors[:, band_idx]

    out_image = os.path.join(output_dir, f"{osm_filename}.tif")
    with rasterio.open(osm_raster_path) as src:
        with rasterio.open(
            out_image,
            "w",
            driver="GTiff",
            count=3,
            height=osm_raster.shape[1],
            width=osm_raster.shape[2],
            dtype=osm_raster.dtype,
            crs=src.crs,
            transform=src.transform,
        ) as dst:
            dst.write(osm_raster)

    # Downscale raster from 5m to 10m resolution after filling missing data
    output_reprojected_path = os.path.join(output_dir, "10m", f"{osm_filename}.tif")
    gdal.Warp(
        output_reprojected_path,
        out_image,
        xRes=TARGET_RESOLUTION,
        yRes=TARGET_RESOLUTION,
        resampleAlg="nearest", # use "lanczos" for smoother boundaries but increase aliasing effect
    )


def fill_missing_osm_data(
    osm_raster_path: str,
    CLC_path: str,
    output_dir: str,
    CLC_color_mapping: dict,
    color_dict: dict,
) -> None:
    """Full processing to fill OSM no data with CLC matching data type.

    Args:
        osm_raster_path: Path to input OSM raster.
        CLC_path: Path to full CLC raster.
        output_dir: Path to output directory.
        CLC_color_mapping: Dictionary containing colors for CLC values.
        color_dict: Dictionary containing color codes for color mapping.
    """
    tmp_dir = os.path.join(output_dir, "tmp")
    output_CLC_dir = os.path.join(output_dir, "CLC_crop")

    clc_crop = extract_CLC_crop(osm_raster_path, CLC_path, tmp_dir, output_CLC_dir)
    fill_osm_no_data(osm_raster_path, clc_crop, output_dir, CLC_color_mapping, color_dict)
