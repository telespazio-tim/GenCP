import argparse
import glob
import logging
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from .clc_processing import fill_missing_osm_data
from .config import (CLC_color_mapping, DEFAULT_CHIP_SIZE_METERS,
                     HRPipelineConfig, TARGET_RESOLUTION, color_dict,
                     highway_colors, landuse_colors, natural_colors)
from .geo_utils import (create_reference_raster, extract_bbox_from_vector,
                        extract_mgrs_tile, generate_chip_grid, get_stem)
from .image_chips_extractor import data_collector, tile_s2_full_image
from .osm_extractor import (create_osm_rasters_for_chips,
                             create_osm_rasters_for_chips_from_bounds)
from .topography_processing import extract_DEM_crop

logger = logging.getLogger(__name__)


def create_folders(
    input_image: str,
    band: str,
    S2_crop_folder: str,
    output_dir: str,
) -> Tuple[str, str, str, str, str]:
    """Create the output directory structure for a given S2 product.

    Args:
        input_image: Path to the S2 product directory.
        band: S2 band name (e.g., 'TCI', 'B04').
        S2_crop_folder: Name of the S2 crop subfolder.
        output_dir: Root output directory.

    Returns:
        Tuple of (input_image_dir, output_osm_dir, output_osm_filled_dir,
                  img_fold_AB, output_topo_dir).
    """
    product_name = get_stem(input_image)

    input_image_dir = os.path.join(output_dir, product_name, band, S2_crop_folder)
    output_osm_dir = os.path.join(output_dir, product_name, "OSM_5m")
    output_topo_dir = os.path.join(output_dir, product_name, "TOPO")
    output_osm_filled_dir = os.path.join(output_osm_dir, "image_filled")
    img_fold_AB = os.path.join(output_dir, product_name, f"AB_{band}_5m")

    os.makedirs(output_osm_dir, exist_ok=True)
    os.makedirs(output_topo_dir, exist_ok=True)
    os.makedirs(os.path.join(output_osm_dir, "geojson"), exist_ok=True)
    os.makedirs(os.path.join(output_osm_dir, "image"), exist_ok=True)
    os.makedirs(output_osm_filled_dir, exist_ok=True)
    os.makedirs(os.path.join(output_osm_filled_dir, "tmp"), exist_ok=True)
    os.makedirs(os.path.join(output_osm_filled_dir, "CLC_crop"), exist_ok=True)
    os.makedirs(os.path.join(output_osm_filled_dir, "10m"), exist_ok=True)
    os.makedirs(img_fold_AB, exist_ok=True)

    return (input_image_dir, output_osm_dir,
            output_osm_filled_dir, img_fold_AB, output_topo_dir)


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for the HR data preparation pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', type=str, help='Path to directory containing S2 products')
    parser.add_argument('--output_dir', '-o', type=str, help='Path to output directory')
    parser.add_argument('--band', '-b', type=str, default="TCI", help='S2 band to use for dataset creation from [B04, TCI]')
    parser.add_argument('--osm', action='store_true', help='Download OSM data if set')
    parser.add_argument('--gri', action='store_true', help='Download S2 GRI data if set')
    parser.add_argument('--buffer', type=int, default=1000, help='Buffer size to extend S2 GRI crop')
    parser.add_argument('--no_buffer', action='store_true', help='No buffer extension if set')
    parser.add_argument('--clc', type=str, help='Path to CLC raster file')
    parser.add_argument('--no_data_threshold', type=int, help='Percentage of accepted no data for OSM raster')
    parser.add_argument('--rm_no_data', action='store_true', help='Remove OSM raster when no data > threshold if set')
    parser.add_argument('--merge_data', action='store_true', help='Create merged images between S2 crops and OSM rasters if set')
    parser.add_argument('--multiprocess', '-mp', action='store_true', help='Use multi-processing if set')
    parser.add_argument('--topo', action='store_true', help='Create raster chips from topography layer if set to True')
    parser.add_argument('--topo_dir', type=str, help='path to topography data directory')
    parser.add_argument('--highway_threshold', type=int, default=1000, help='Threshold value to filter invisible highway features in S2 B04')
    # AOI / full-tile mode arguments
    parser.add_argument('--aoi', type=str, help='Path to a GeoJSON or shapefile defining the AOI (replaces S2 input)')
    parser.add_argument('--full_tile', action='store_true',
                        help='Tile each S2 image\'s full footprint into chips instead of using GRI GCPs')
    parser.add_argument('--chip_size', type=float, default=DEFAULT_CHIP_SIZE_METERS,
                        help='Chip side length in meters for AOI / full-tile mode (default: 2560 = 256px at 10m)')
    parser.add_argument('--osm_batch_max_deg', type=float, default=0.2,
                        help='Max bbox extent (deg) per Overpass batch. Lower = less memory, '
                             'more HTTP calls. <=0 disables clustering (default: 0.2)')

    args = parser.parse_args()
    return args


def merge_s2_osm_pairs(
    input_image_dir: str,
    osm_filled_dir: str,
    output_dir: str,
    band: str,
) -> None:
    """Join S2 and OSM images side-by-side and save in the AB folder.

    Args:
        input_image_dir: Directory containing S2 crop images.
        osm_filled_dir: Directory containing filled OSM rasters.
        output_dir: Directory to write merged AB images.
        band: S2 band name (determines grayscale normalisation for B04).
    """
    img_list = [f for f in os.listdir(input_image_dir) if f.endswith('.tif')]
    for img_name in img_list:
        path_A = os.path.join(input_image_dir, img_name)
        path_B = os.path.join(osm_filled_dir, "10m", img_name)
        if not (os.path.isfile(path_A) and os.path.isfile(path_B)):
            continue

        path_AB = os.path.join(output_dir, img_name)
        if band == "B04":
            im_A_gray = cv2.imread(path_A, cv2.IMREAD_UNCHANGED)
            im_A_norm = cv2.normalize(
                im_A_gray, None, alpha=0, beta=255,
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U,
            )
            im_A = cv2.cvtColor(im_A_norm, cv2.COLOR_GRAY2RGB)
        else:
            im_A = cv2.imread(path_A, 1)

        im_B = cv2.imread(path_B, 1)
        im_AB = np.concatenate([im_A, im_B], 1)
        cv2.imwrite(path_AB, im_AB)

    logger.info("Merged %d image pairs into %s", len(img_list), output_dir)


def create_aoi_folders(
    aoi_name: str,
    output_dir: str,
) -> Tuple[str, str, str]:
    """Create the output directory structure for AOI mode.

    Args:
        aoi_name: Name derived from the AOI file (used as product folder).
        output_dir: Root output directory.

    Returns:
        Tuple of (output_osm_dir, output_osm_filled_dir, ref_raster_dir).
    """
    output_osm_dir = os.path.join(output_dir, aoi_name, "OSM_5m")
    output_osm_filled_dir = os.path.join(output_osm_dir, "image_filled")
    ref_raster_dir = os.path.join(output_dir, aoi_name, "ref_rasters")

    os.makedirs(os.path.join(output_osm_dir, "geojson"), exist_ok=True)
    os.makedirs(os.path.join(output_osm_dir, "image"), exist_ok=True)
    os.makedirs(output_osm_filled_dir, exist_ok=True)
    os.makedirs(os.path.join(output_osm_filled_dir, "tmp"), exist_ok=True)
    os.makedirs(os.path.join(output_osm_filled_dir, "CLC_crop"), exist_ok=True)
    os.makedirs(os.path.join(output_osm_filled_dir, "10m"), exist_ok=True)
    os.makedirs(ref_raster_dir, exist_ok=True)

    return output_osm_dir, output_osm_filled_dir, ref_raster_dir


def run_aoi_pipeline(config: HRPipelineConfig) -> None:
    """Run the AOI-based pipeline: tile the AOI into chips and generate OSM rasters.

    Args:
        config: Pipeline configuration (must have aoi_path set).
    """
    from pyproj import Transformer

    aoi_name = get_stem(config.aoi_path)
    output_osm_dir, output_osm_filled_dir, ref_raster_dir = create_aoi_folders(
        aoi_name, str(config.output_dir),
    )

    # 1. Generate chip grid from AOI
    chips = generate_chip_grid(
        str(config.aoi_path), config.chip_size, TARGET_RESOLUTION,
    )
    if not chips:
        logger.warning("No chips generated from AOI %s", config.aoi_path)
        return

    # 2. Reference rasters + per-chip bbox in EPSG:4326
    ref_paths: dict = {}
    for chip in chips:
        try:
            ref_path = os.path.join(ref_raster_dir, f"{chip['name']}.tif")
            create_reference_raster(chip, ref_path)
            ref_paths[chip['name']] = ref_path

            left, bottom, right, top = chip["bounds"]
            transformer = Transformer.from_crs(chip["crs"], "EPSG:4326", always_xy=True)
            lon_min, lat_min = transformer.transform(left, bottom)
            lon_max, lat_max = transformer.transform(right, top)
            chip['bbox_4326'] = (lon_min, lat_min, lon_max, lat_max)  # canonical (w, s, e, n)
        except Exception:
            logger.exception("Failed to prepare chip: %s", chip['name'])

    chips_ready = [c for c in chips if 'bbox_4326' in c and c['name'] in ref_paths]

    # 3. One Overpass call covers all chips, then per-chip subset + rasterize
    create_osm_rasters_for_chips_from_bounds(
        chips_ready, ref_paths, output_osm_dir,
        highway_colors, natural_colors, landuse_colors, color_dict,
        batch_max_deg=config.osm_batch_max_deg,
    )

    # 4. CLC fill (optional)
    if config.clc_path:
        for chip in chips_ready:
            osm_image_path = os.path.join(output_osm_dir, "image", f"{chip['name']}.tif")
            if os.path.isfile(osm_image_path):
                try:
                    fill_missing_osm_data(
                        osm_image_path, str(config.clc_path),
                        output_osm_filled_dir, CLC_color_mapping, color_dict,
                    )
                except Exception:
                    logger.exception("CLC fill failed for chip: %s", chip['name'])

    logger.info("AOI pipeline completed: %d chips in %s", len(chips_ready), output_osm_dir)


def _run_osm_topo_merge_steps(
    config: HRPipelineConfig,
    S2_path: str,
    input_image_dir: str,
    output_osm_dir: str,
    output_osm_filled_dir: str,
    img_fold_AB: str,
    output_topo_dir: str,
) -> None:
    """Run OSM rasterization, CLC fill, topography crop, and AB merge.

    Shared post-chip-extraction steps for both ``run_s2_pipeline`` and
    ``run_s2_full_tile_pipeline``: chips already exist on disk in
    ``input_image_dir``; this turns them into OSM rasters (and optionally
    CLC-filled / topo-paired / AB-merged outputs).
    """
    if config.download_osm:
        existing_osm_names = {
            get_stem(p) for p in glob.glob(os.path.join(output_osm_dir, "image", "*.tif"))
        }
        all_chips = glob.glob(os.path.join(input_image_dir, "*.tif"))
        pending_chips = [
            p for p in all_chips if get_stem(p) not in existing_osm_names
        ]

        if not pending_chips:
            logger.info("Skipping OSM for %s: all %d rasters already exist in %s",
                        Path(S2_path).name, len(existing_osm_names), output_osm_dir)
        else:
            if existing_osm_names:
                logger.info("Resuming OSM for %s: %d of %d chips still pending",
                            Path(S2_path).name, len(pending_chips), len(all_chips))

            # One Overpass call for all pending chips of this S2 product, then
            # per-chip spatial subset + in-memory rasterization. The previous
            # per-chip Pool issued N HTTP requests; now it's one.
            create_osm_rasters_for_chips(
                pending_chips, output_osm_dir, highway_colors, natural_colors,
                landuse_colors, color_dict, config.highway_threshold,
                batch_max_deg=config.osm_batch_max_deg,
            )

        if config.clc_path:
            existing_filled = {
                get_stem(p) for p in glob.glob(os.path.join(output_osm_filled_dir, "10m", "*.tif"))
            }
            all_osm = glob.glob(os.path.join(output_osm_dir, "image", "*.tif"))
            pending_osm = [
                p for p in all_osm if get_stem(p) not in existing_filled
            ]

            if not pending_osm:
                logger.info("Skipping OSM fill for %s: all %d filled rasters already exist in %s",
                            Path(S2_path).name, len(existing_filled), output_osm_filled_dir)
            else:
                if existing_filled:
                    logger.info("Resuming OSM fill for %s: %d of %d rasters still pending",
                                Path(S2_path).name, len(pending_osm), len(all_osm))

                if config.use_multiprocess:
                    parameters = [
                        (osm_image_path, str(config.clc_path), output_osm_filled_dir,
                            CLC_color_mapping, color_dict)
                        for osm_image_path in pending_osm
                    ]
                    with Pool() as p:
                        p.starmap(fill_missing_osm_data, parameters)
                else:
                    for osm_image_path in pending_osm:
                        fill_missing_osm_data(osm_image_path, str(config.clc_path), output_osm_filled_dir, CLC_color_mapping, color_dict)

    if config.extract_topo:
        mgrs_tile = extract_mgrs_tile(Path(S2_path).name)
        topo_matches = glob.glob(os.path.join(str(config.topo_dir), f"{mgrs_tile}*.tif"))
        if not topo_matches:
            logger.warning("No topography file found for tile %s in %s", mgrs_tile, config.topo_dir)
        else:
            topo_path = topo_matches[0]
            for image_path in glob.glob(os.path.join(output_osm_filled_dir, "10m", "*.tif")):
                extract_DEM_crop(image_path, topo_path, output_topo_dir)

    if config.merge_data:
        merge_s2_osm_pairs(
            input_image_dir, output_osm_filled_dir,
            img_fold_AB, config.band,
        )


def run_s2_pipeline(config: HRPipelineConfig) -> None:
    """Run the S2-based pipeline (original behaviour).

    Args:
        config: Pipeline configuration.
    """
    for S2_path in glob.glob(os.path.join(str(config.input_dir), "S2*")):

        try:
            (input_image_dir, output_osm_dir,
             output_osm_filled_dir, img_fold_AB,
             output_topo_dir) = create_folders(
                S2_path, config.band, config.s2_crop_folder, str(config.output_dir)
            )

            # 1. Get S2 training images
            if config.download_gri:
                existing_crops = glob.glob(os.path.join(input_image_dir, "*.tif"))
                if existing_crops:
                    logger.info("Skipping GRI for %s: %d crops already exist in %s",
                                Path(S2_path).name, len(existing_crops), input_image_dir)
                else:
                    data_collector(S2_path, config.band, config.apply_buffer, config.buffer_size, str(config.output_dir))

            # 2-5. OSM rasterization, CLC fill, topo crop, AB merge
            _run_osm_topo_merge_steps(
                config, S2_path, input_image_dir, output_osm_dir,
                output_osm_filled_dir, img_fold_AB, output_topo_dir,
            )

        except Exception:
            logger.exception("Failed to process product: %s", S2_path)
            continue


def run_s2_full_tile_pipeline(config: HRPipelineConfig) -> None:
    """Tile each S2 image's full footprint into chips, then run OSM steps.

    Mirrors ``run_s2_pipeline`` but replaces the GRI-based chip extraction
    with a regular grid covering the entire S2 image footprint. The B04 band
    is also tiled (when not the primary band) so the highway-visibility
    filter in ``create_osm_raster`` still works.

    Args:
        config: Pipeline configuration.
    """
    for S2_path in glob.glob(os.path.join(str(config.input_dir), "S2*")):

        try:
            (input_image_dir, output_osm_dir,
             output_osm_filled_dir, img_fold_AB,
             output_topo_dir) = create_folders(
                S2_path, config.band, config.s2_crop_folder, str(config.output_dir)
            )

            # 1. Tile the full S2 footprint into chips
            existing_crops = glob.glob(os.path.join(input_image_dir, "*.tif"))
            if existing_crops:
                logger.info("Skipping tile extraction for %s: %d crops already exist in %s",
                            Path(S2_path).name, len(existing_crops), input_image_dir)
            else:
                tile_s2_full_image(
                    S2_path, config.band, config.chip_size,
                    str(config.output_dir), apply_buffer=config.apply_buffer,
                    use_multiprocess=config.use_multiprocess,
                )
                # B04 chips are needed for the highway-visibility filter
                if config.band != "B04":
                    tile_s2_full_image(
                        S2_path, "B04", config.chip_size,
                        str(config.output_dir), apply_buffer=config.apply_buffer,
                        use_multiprocess=config.use_multiprocess,
                    )

            # 2-5. OSM rasterization, CLC fill, topo crop, AB merge
            _run_osm_topo_merge_steps(
                config, S2_path, input_image_dir, output_osm_dir,
                output_osm_filled_dir, img_fold_AB, output_topo_dir,
            )

        except Exception:
            logger.exception("Failed to process product: %s", S2_path)
            continue


def main() -> None:
    """Run the HR data preparation pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = get_args()
    config = HRPipelineConfig.from_args(args)

    if config.is_aoi_mode:
        logger.info("Running in AOI mode from %s", config.aoi_path)
        run_aoi_pipeline(config)
    elif config.is_full_tile_mode:
        logger.info("Running in full-tile mode (chip_size=%.0fm) over %s",
                    config.chip_size, config.input_dir)
        run_s2_full_tile_pipeline(config)
    else:
        run_s2_pipeline(config)


if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    logging.getLogger(__name__).info("Pipeline completed in %.1f seconds", elapsed)
