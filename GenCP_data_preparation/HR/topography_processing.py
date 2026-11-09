"""DEM / topography crop extraction for the HR data preparation pipeline."""

import os

import rasterio

from .geo_utils import crop_raster_to_bounds, get_stem, validate_file_exists


def extract_DEM_crop(image_path: str, dem_path: str, output_dir: str) -> str:
    """Extract DEM raster matching image chips bounds.

    Args:
        image_path: Path to input image chips.
        dem_path: Path to DEM raster.
        output_dir: Path to output directory.

    Returns:
        Path to the output cropped DEM raster. It keeps the chip's file name so
        that ``create_train_test_split.split_topography`` can pair it with the
        A/B chips by stem.
    """
    validate_file_exists(image_path, "input image chip")
    validate_file_exists(dem_path, "DEM raster")

    img_filename = get_stem(image_path)

    with rasterio.open(image_path) as src:
        bounds = src.bounds
        src_crs = src.crs

    output_path = os.path.join(output_dir, f"{img_filename}.tif")
    crop_raster_to_bounds(dem_path, bounds, output_path, src_crs=src_crs)
    return output_path
