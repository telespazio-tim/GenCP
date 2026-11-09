"""Sentinel-2 image chip extraction and GRI data collection."""

import glob
import logging
import math
import os
import tarfile
from multiprocessing import Pool
from typing import Optional

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
import rasterio.mask
import requests
from osgeo import gdal
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from shapely.geometry import box

from .config import (
    GRI_BASE_URL,
    QUICKLOOK_QUANTIFICATION,
    QUICKLOOK_SITE_OVERRIDES,
    QUICKLOOK_SRC_MAX,
    QUICKLOOK_SRC_MAX_SWIR,
    TARGET_RESOLUTION,
)
from ..common.tile_filters import is_all_black
from .geo_utils import (
    extract_band_basename,
    extract_mgrs_tile,
    generate_chip_grid_from_bounds,
    get_stem,
    validate_dir_exists,
    validate_file_exists,
)

logger = logging.getLogger(__name__)


def extend_gri_chips(image_path: str, buffer_size: int, out_image: str) -> None:
    """Extend raster with nodata to create bigger raster.

    Args:
        image_path: Path to input raster.
        buffer_size: Buffer size in meters for raster extension.
        out_image: Path to output raster.
    """
    with rasterio.open(image_path) as src:
        bounds = src.bounds

    bbox = box(*bounds)
    extended_bbox = bbox.buffer(buffer_size, cap_style=3, join_style=2)

    coor_list = list(extended_bbox.exterior.coords)
    x_coordinates, y_coordinates = zip(*coor_list)
    new_coord = [min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)]

    ds = gdal.Open(image_path)
    ds = gdal.Warp(out_image, ds, outputBounds=new_coord)
    ds = None


def extract_S2_crops(S2_path: str, GRI_path: str, out_image: str) -> None:
    """Extract small crops from S2 image corresponding to GRI GCPs.

    Args:
        S2_path: Path to input S2 band.
        GRI_path: Path to GRI image chip.
        out_image: Path to output cropped image.
    """
    with rasterio.open(GRI_path) as src:
        GRI_bounds = src.bounds

    mask = [box(*GRI_bounds)]
    with rasterio.open(S2_path) as src:
        image_crop, transformed = rasterio.mask.mask(src, mask, crop=True)
        out_profile = src.profile.copy()

    no_data = image_crop[image_crop == 0]
    if len(no_data) == 0:
        out_profile.update({
            'width': image_crop.shape[2],
            'height': image_crop.shape[1],
            'transform': transformed,
        })
        with rasterio.open(out_image, 'w', **out_profile) as dst:
            dst.write(image_crop)


def select_gri_chips(S2_path: str, GRI_path: str) -> bool:
    """Select GRI image chips that are located inside the S2 image.

    Args:
        S2_path: Path to input S2 band.
        GRI_path: Path to GRI image chip.

    Returns:
        True if GRI chip is fully contained within S2 image bounds.
    """
    with rasterio.open(GRI_path) as gri:
        GRI_bounds = gri.bounds

    with rasterio.open(S2_path) as src:
        S2_bounds = src.bounds

    return box(*S2_bounds).contains(box(*GRI_bounds))


def _format_gri_coordinates(lat: int, lon: int) -> tuple:
    """Format latitude and longitude for GRI tile naming.

    Args:
        lat: Latitude as integer (e.g., 43, -12).
        lon: Longitude as integer (e.g., 1, -120).

    Returns:
        Tuple of (lat_str, lon_str) formatted for GRI URLs (e.g., 'N43', 'W012').
    """
    lat_str = ("S" if lat < 0 else "N") + f"{abs(lat):02d}"
    lon_str = ("W" if lon < 0 else "E") + f"{abs(lon):03d}"
    return lat_str, lon_str


def gri_cells(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> list:
    """List the 1-degree GRI cells overlapping a lon/lat bounding box.

    A cell is named after its south-west corner, so a point at lon -1.5
    belongs to cell -2 (W002): ``floor`` is used, not ``int``, and every cell
    between the min and max corners is listed (a S2 tile can span 3 cells).

    Returns:
        Sorted list of (lat, lon) integer pairs.
    """
    lats = range(math.floor(lat_min), math.floor(lat_max) + 1)
    lons = range(math.floor(lon_min), math.floor(lon_max) + 1)
    return [(lat, lon) for lat in lats for lon in lons]


def download_gri_chips(S2_path: str, output_dir: str) -> None:
    """Download GRI GCPs overlapping S2 image.

    Args:
        S2_path: Path to input S2 band.
        output_dir: Path to output directory.
    """
    with rasterio.open(S2_path) as src:
        S2_bounds = src.bounds

    dst_crs = 'EPSG:4326'
    lon_min, lat_min, lon_max, lat_max = transform_bounds(src.crs, dst_crs, *S2_bounds)

    for lat, lon in gri_cells(lon_min, lat_min, lon_max, lat_max):
        lat_str, lon_str = _format_gri_coordinates(lat, lon)

        download_url = f"{GRI_BASE_URL}/{lat_str}/{lat_str}{lon_str}_L1C.tar.gz"
        try:
            res = requests.get(url=download_url, timeout=20, allow_redirects=False, stream=True)
            if res.status_code == 200:
                block_size = 10 * 1024
                file_name = f"test_download_{lat_str}_{lon_str}.tar"
                with open(file_name, mode="wb") as file:
                    for chunk in res.iter_content(chunk_size=block_size):
                        file.write(chunk)

                with tarfile.open(file_name) as tar:
                    tar.extractall(output_dir, filter='data')

                os.remove(file_name)

        except requests.exceptions.HTTPError as err:
            logger.warning("Failed to download GRI chip %s%s: %s", lat_str, lon_str, err)
            raise


def quicklook(
    product_dir: str,
    bands: Optional[list] = None,
    quality: int = 95,
    xRes: float = TARGET_RESOLUTION,
    yRes: float = TARGET_RESOLUTION,
    out_format: str = 'GTIFF',
    creationOptions: Optional[list] = None,
    offset: int = 0,
) -> Optional[str]:
    """Generate a quicklook TCI image from individual S2 bands.

    Args:
        product_dir: S2 product directory.
        bands: List of 3 band names for [R, G, B].
        quality: JPEG quality (for non-GTIFF output).
        xRes: X resolution.
        yRes: Y resolution.
        out_format: Output format.
        creationOptions: GDAL creation options.
        offset: Radiometric offset.

    Returns:
        Output file path if successful, otherwise None.
    """
    validate_dir_exists(product_dir, "S2 product directory")

    if bands is None:
        bands = ["B04", "B03", "B02"]

    red = glob.glob(os.path.join(product_dir, "GRANULE", "*", "IMG_DATA", "*B04_10m.TIF"))[0]
    green = glob.glob(os.path.join(product_dir, "GRANULE", "*", "IMG_DATA", "*B03_10m.TIF"))[0]
    blue = glob.glob(os.path.join(product_dir, "GRANULE", "*", "IMG_DATA", "*B02_10m.TIF"))[0]

    images = {"B04": red, "B03": green, "B02": blue}

    band_dir_path = os.path.dirname(red)
    band_basename = extract_band_basename(red)
    qlpath = os.path.join(band_dir_path, f"{band_basename}_TCI_10m.TIF")

    imagefiles = []

    # bands for rgb
    for band in bands:
        if band not in images.keys():
            logger.error("Bands %s not available for quicklook", bands)
            return None
        else:
            imagefiles.append(images[band])

    # create output directory if it does not exist
    qldir = os.path.dirname(qlpath)
    os.makedirs(qldir, exist_ok=True)

    # Grayscale or RGB
    if len(bands) == 1:
        band_list = [1]
    else:
        band_list = [1, 2, 3]

    # create single vrt with B4 B3 B2
    vrtpath = qlpath + '.vrt'

    gdal.BuildVRT(vrtpath, imagefiles, separate=True)
    logger.debug("VRT saved to: %s", vrtpath)

    # Remove nodata attribute
    vrt = gdal.Open(vrtpath, gdal.GA_Update)
    for i in band_list:
        vrt.GetRasterBand(i).DeleteNoDataValue()
    del vrt

    # Radiometric scaling
    src_min = 0
    src_max = QUICKLOOK_SRC_MAX
    dst_min = 0
    dst_max = 255
    if bands == ["B12", "B11", "B8A"]:
        src_max = QUICKLOOK_SRC_MAX_SWIR

    # Site-specific overrides
    for site_id, overrides in QUICKLOOK_SITE_OVERRIDES.items():
        if site_id in product_dir:
            src_max = overrides.get("default", src_max)
            if bands == ["B12", "B11", "B8A"]:
                src_max = overrides.get("swir", src_max)

    scale = [[src_min + offset, src_max + offset, dst_min, dst_max]]

    # GDAL translate
    if out_format == 'GTIFF':
        create_options = creationOptions
    else:
        create_options = [f'QUALITY={quality}'] if creationOptions is None else [f'QUALITY={quality}'] + creationOptions

    dataset = gdal.Translate(qlpath, vrtpath, xRes=xRes, yRes=yRes, resampleAlg='bilinear', bandList=band_list,
                   outputType=gdal.GDT_Byte, format=out_format, creationOptions=create_options,
                   scaleParams=scale)

    logger.info("Quicklook saved to: %s", qlpath)

    scaling = (src_max - src_min) / QUICKLOOK_QUANTIFICATION / (dst_max - dst_min)

    try:
        for i in band_list:
            dataset.GetRasterBand(i).SetScale(scaling)
            dataset.GetRasterBand(i).SetOffset(0)
            dataset.GetRasterBand(i).DeleteNoDataValue()

        logger.debug("Scale and offset metadata written to quicklook")
        dataset = None

    except Exception as e:
        logger.error("Failed to update quicklook metadata: %s", e, exc_info=True)
        raise

    # clean
    os.remove(vrtpath)

    return qlpath


def is_in_cloud_mask(cloud_mask_path: str, GRI_path: str) -> bool:
    """Check if cloud mask overlaps image.

    Args:
        cloud_mask_path: Path to cloud mask GML file.
        GRI_path: Path to GRI image chip.

    Returns:
        True if cloud mask overlaps GRI image chip.
    """
    in_cloud = False

    try:
        with rasterio.open(GRI_path) as src:
            GRI_bounds = src.bounds

        GRI_poly = box(*GRI_bounds)
        cloud_mask_gpd = gpd.read_file(cloud_mask_path, driver="GML")

        for cloud_poly in list(cloud_mask_gpd["geometry"]):
            in_cloud = GRI_poly.intersects(cloud_poly)
            if in_cloud:
                break
    except Exception as e:
        logger.warning("Cloud mask check failed: %s", e)

    return in_cloud


# ---------------------------------------------------------------------------
# Unified data collector (replaces data_collector_from_S2 / _from_S2L)
# ---------------------------------------------------------------------------

def _resolve_band_path(S2_path: str, band: str, source_type: str) -> str:
    """Resolve the band file path based on source type and band."""
    if source_type == "S2":
        pattern = os.path.join(S2_path, 'GRANULE', '*', 'IMG_DATA', 'R10m', f'*_{band}_10m.jp2')
    elif band == "B08":
        pattern = os.path.join(S2_path, 'GRANULE', '*', 'IMG_DATA', 'NATIVE', f'*{band}_10m.TIF')
    else:
        pattern = os.path.join(S2_path, 'GRANULE', '*', 'IMG_DATA', f'*{band}_10m.TIF')

    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No band file found matching pattern: {pattern}")
    return matches[0]


def _find_cloud_mask(S2_path: str) -> Optional[str]:
    """Find cloud mask GML file if it exists."""
    masks = glob.glob(os.path.join(S2_path, 'GRANULE', '*', 'QI_DATA', 'MSK_CLOUDS_B00.gml'))
    return masks[0] if masks else None


def _should_skip_cloud(cloud_mask_path: Optional[str], reference_path: str) -> bool:
    """Return True if the reference image is cloudy and should be skipped."""
    if cloud_mask_path is None:
        return False
    return is_in_cloud_mask(cloud_mask_path, reference_path)


def data_collector(
    S2_path: str,
    band: str,
    apply_buffer: bool,
    buffer_size: int,
    output_dir: str,
    source_type: str = "S2",
    apply_cloud_mask: bool = False,
) -> None:
    """Collect and extract small S2/S2L crops corresponding to GRI GCP extents.

    Unified function replacing data_collector_from_S2 and data_collector_from_S2L.

    Args:
        S2_path: Path to input S2/S2L folder.
        band: S2 band in ['B04', 'B08', 'TCI'].
        apply_buffer: Extend GRI image chips extents if set to True.
        buffer_size: Buffer size in meters for raster extension.
        output_dir: Path to output directory.
        source_type: "S2" for standard L2A products, "S2L" for Sen2Like products.
        apply_cloud_mask: Apply cloud mask filter (S2L only).
    """
    validate_dir_exists(S2_path, "S2 product directory")

    # 1. Optionally generate TCI quicklook (S2L only)
    if source_type == "S2L" and band == "TCI":
        quicklook(S2_path)

    # 2. Resolve band path
    band_path = _resolve_band_path(S2_path, band, source_type)

    # 3. Resolve cloud mask (S2L only)
    cloud_mask_path = None
    if source_type == "S2L" and apply_cloud_mask:
        cloud_mask_path = _find_cloud_mask(S2_path)

    pdt_name = get_stem(S2_path)

    # 4. Download GRI images chips overlapping the S2 image
    os.makedirs(output_dir, exist_ok=True)

    output_gri_folder = os.path.join(output_dir, pdt_name, "GRI")
    download_gri_chips(band_path, output_gri_folder)

    # 5. Extract the S2 crops and save dataset
    s2_band_dir = os.path.join(output_dir, pdt_name, band)
    os.makedirs(s2_band_dir, exist_ok=True)

    if apply_buffer:
        gri_extended_dir = os.path.join(output_dir, pdt_name, "GRI_extended")
        s2_crop_dir = os.path.join(s2_band_dir, "S2_crop_extended")
        os.makedirs(gri_extended_dir, exist_ok=True)
    else:
        s2_crop_dir = os.path.join(s2_band_dir, "S2_crop")

    os.makedirs(s2_crop_dir, exist_ok=True)

    for image_chips_path in glob.glob(os.path.join(output_gri_folder, '*', 'L1C_chips', '*00.TIF'), recursive=True):

        # check if image is inside S2 image
        if select_gri_chips(band_path, image_chips_path):
            chip_name = get_stem(image_chips_path)

            if apply_buffer:
                buffered_image = os.path.join(gri_extended_dir, f"{chip_name}.tif")
                extend_gri_chips(image_chips_path, buffer_size, buffered_image)

                if not select_gri_chips(band_path, buffered_image):
                    os.remove(buffered_image)
                else:
                    if _should_skip_cloud(cloud_mask_path, buffered_image):
                        continue
                    cropped_image = os.path.join(s2_crop_dir, f"{chip_name}.tif")
                    extract_S2_crops(band_path, buffered_image, cropped_image)

            else:
                if _should_skip_cloud(cloud_mask_path, image_chips_path):
                    continue
                cropped_image = os.path.join(s2_crop_dir, f"{chip_name}.tif")
                extract_S2_crops(band_path, image_chips_path, cropped_image)


# ---------------------------------------------------------------------------
# Full-image tiling (no GRI)
# ---------------------------------------------------------------------------


def _crop_chip_with_window(
    src: "rasterio.io.DatasetReader",
    chip: dict,
    output_path: str,
) -> bool:
    """Read a chip from an already-open S2 band raster and write a GeoTIFF.

    Uses ``windows.from_bounds`` against the dataset's transform so a chip
    that falls fully inside the raster is fetched as a single pixel window —
    no spatial intersection, no per-chip ``rasterio.open``.

    Args:
        src: Open rasterio dataset for the S2 band (kept open by the caller).
        chip: Chip dict from ``generate_chip_grid_from_bounds`` (bounds in the
            band's CRS).
        output_path: Output GeoTIFF path.

    Returns:
        True if a chip was written; False if the chip falls partially or
        fully outside the raster, or contains only no-data pixels.
    """
    left, bottom, right, top = chip["bounds"]
    window = from_bounds(left, bottom, right, top, src.transform)

    col_off = int(round(window.col_off))
    row_off = int(round(window.row_off))
    width = int(round(window.width))
    height = int(round(window.height))

    if (col_off < 0 or row_off < 0
            or col_off + width > src.width
            or row_off + height > src.height):
        return False

    if width != chip["width"] or height != chip["height"]:
        return False

    read_window = Window(col_off, row_off, width, height)
    image_crop = src.read(window=read_window)

    if is_all_black(image_crop):
        return False

    out_profile = src.profile.copy()
    out_profile.update({
        "width": width,
        "height": height,
        "transform": src.window_transform(read_window),
    })
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(image_crop)
    return True


_chip_worker_src: Optional["rasterio.io.DatasetReader"] = None


def _init_chip_worker(band_path: str) -> None:
    """Pool initializer: open the S2 band raster once per worker process."""
    global _chip_worker_src
    _chip_worker_src = rasterio.open(band_path)


def _chip_worker(chip: dict, output_path: str) -> bool:
    """Pool worker: crop one chip using the process-local open dataset."""
    return _crop_chip_with_window(_chip_worker_src, chip, output_path)


def tile_s2_full_image(
    S2_path: str,
    band: str,
    chip_size_meters: float,
    output_dir: str,
    apply_buffer: bool = True,
    source_type: str = "S2",
    use_multiprocess: bool = False,
) -> int:
    """Tile a full S2 image footprint into chips and save them as GeoTIFFs.

    Mirrors the output layout used by ``data_collector`` so that downstream
    OSM/CLC/topo/merge steps work unchanged.

    Args:
        S2_path: Path to the S2 product directory.
        band: S2 band name in {"B04", "B08", "TCI"}.
        chip_size_meters: Side length of each chip in meters.
        output_dir: Root output directory.
        apply_buffer: If True, write into ``S2_crop_extended/`` (matches the
            buffered GRI layout); otherwise into ``S2_crop/``.
        source_type: "S2" for standard L2A, "S2L" for Sen2Like.
        use_multiprocess: Process chips in a worker pool. Each worker opens
            the S2 band once (via the Pool initializer), then crops its
            assigned chips with windowed reads.

    Chips are named ``T<MGRS>_tile_<col>_<row>`` (e.g. ``T31TCJ_tile_0003_0012``)
    so that chips of different S2 tiles do not collide when the train/test
    split gathers all products, and so that summer and winter products of the
    same tile give the same names.

    Returns:
        Number of chips successfully written.
    """
    validate_dir_exists(S2_path, "S2 product directory")

    band_path = _resolve_band_path(S2_path, band, source_type)
    pdt_name = get_stem(S2_path)

    s2_band_dir = os.path.join(output_dir, pdt_name, band)
    crop_subfolder = "S2_crop_extended" if apply_buffer else "S2_crop"
    s2_crop_dir = os.path.join(s2_band_dir, crop_subfolder)
    os.makedirs(s2_crop_dir, exist_ok=True)

    with rasterio.open(band_path) as src:
        bounds = tuple(src.bounds)
        crs = pyproj.CRS(src.crs)

    try:
        name_prefix = f"T{extract_mgrs_tile(pdt_name)}_tile"
    except ValueError:
        name_prefix = f"{pdt_name}_tile"
        logger.warning(
            "No MGRS tile in product name %s: chips are prefixed with the product name",
            pdt_name,
        )
    chips = generate_chip_grid_from_bounds(
        bounds, crs, chip_size_meters, TARGET_RESOLUTION, name_prefix=name_prefix,
    )

    chip_jobs = [
        (chip, os.path.join(s2_crop_dir, f"{chip['name']}.tif"))
        for chip in chips
    ]

    if use_multiprocess and len(chip_jobs) > 1:
        with Pool(initializer=_init_chip_worker, initargs=(band_path,)) as pool:
            results = pool.starmap(_chip_worker, chip_jobs)
        written = sum(1 for r in results if r)
    else:
        with rasterio.open(band_path) as src:
            written = sum(
                1 for chip, out_path in chip_jobs
                if _crop_chip_with_window(src, chip, out_path)
            )

    logger.info(
        "Tiled %s (%s): %d/%d chips written to %s",
        pdt_name, band, written, len(chips), s2_crop_dir,
    )
    return written
