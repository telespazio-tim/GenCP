# HR OSM Dataset Preparation

This module was developed within the GenCP project to prepare paired HR (high-resolution) Sentinel-2 image chips and corresponding OSM raster masks for Generative AI-based image processing workflows.

Unlike the VHR pipeline (which renders styled masks with headless QGIS), the HR pipeline is a self-contained Python package: OSM features are coloured from a fixed palette and rasterized with GDAL/OGR, so **no QGIS installation is required**. It also adds Sentinel-2-specific steps — GRI (Ground Reference Image) georeferencing, CORINE Land Cover (CLC) gap-filling, DEM/topography crops, and seasonal (summer/winter) datasets.

The pipeline is run as a Python module from the repository root:

```bash
python -m GenCP_data_preparation.HR.dataset_creation --input_dir <S2_products> --output_dir <out> [options]
```

---

## Repository contents

```text
GenCP_data_preparation/HR/
├── dataset_creation.py                  # Pipeline entry point (CLI, mode dispatch)
├── config.py                            # Colour dicts, named constants, HRPipelineConfig
├── image_chips_extractor.py             # S2 chip extraction + GRI download / full-tile tiling
├── osm_extractor.py                     # OSM download, styling and GDAL rasterization
├── clc_processing.py                    # CORINE Land Cover gap-filling of OSM rasters
├── topography_processing.py             # DEM / topography crop extraction
├── geo_utils.py                         # Path, MGRS and geospatial helpers
├── create_train_test_split.py           # Post-processing: build train/test split
├── create_seasonal_train_test_split.py  # Post-processing: seasonal (summer/winter) split
├── single_class_tiles.py                # Post-processing: drop single-class OSM tiles
└── README.md
```

Resolution-agnostic plumbing shared with the VHR pipeline (bounding boxes, UTM
estimation, GeoTIFF IO, tiling, tile filters, the osmnx download wrapper, and the
config base) lives in `GenCP_data_preparation/common/`.

---

## Installation

The HR pipeline uses the shared geospatial stack (no QGIS). A conda environment can be created as follows:

```bash
conda create -n gencp-dataprep -c conda-forge python=3.10 \
    numpy geopandas rasterio gdal pyproj shapely "osmnx>=2.0" \
    opencv pillow requests geojson -y
conda activate gencp-dataprep
```

The same list, with the versions the pipeline was tested with, is in [requirements.txt](requirements.txt):

```bash
conda create -n gencp-dataprep -c conda-forge python=3.10 -y
conda activate gencp-dataprep
conda install -c conda-forge --file GenCP_data_preparation/HR/requirements.txt -y
```

> OSM download requires **osmnx >= 2.0** (the pipelines were standardized on the
> osmnx 2.x API and a canonical `(west, south, east, north)` bounding-box order).

---

## Pipeline modes

`dataset_creation.py` selects one of three mutually exclusive modes from the arguments:

| Mode | Trigger | Description |
|---|---|---|
| **S2 (GRI)** | *default* | For each S2 product, download GRI reference chips and cut co-registered S2 crops, then build OSM (+ optional CLC / topo / AB) for each chip. |
| **Full tile** | `--full_tile` | Tile each S2 image's full footprint into a regular grid of chips instead of using GRI chips, then run the same OSM/CLC/topo/merge steps. |
| **AOI** | `--aoi <vector>` | Generate a chip grid over a vector AOI (GeoJSON/shapefile) and build OSM rasters only — no satellite image. |

### Example commands

```bash
# S2 GRI mode: download GRI + OSM, CLC-fill, and write side-by-side AB pairs
python -m GenCP_data_preparation.HR.dataset_creation \
    -i /data/S2_products -o /data/out --band TCI --gri --osm \
    --clc /data/CLC.tif --merge_data --multiprocess

# Full-tile mode over the whole S2 footprint
python -m GenCP_data_preparation.HR.dataset_creation \
    -i /data/S2_products -o /data/out --band B04 --full_tile --osm

# AOI mode (no satellite image)
python -m GenCP_data_preparation.HR.dataset_creation \
    --aoi /data/aoi.geojson -o /data/out --osm --clc /data/CLC.tif
```

---

## Command-line options

| Option | Default | Purpose |
|---|---|---|
| `--input_dir`, `-i` | – | Directory containing Sentinel-2 products |
| `--output_dir`, `-o` | – | Output directory |
| `--band`, `-b` | `TCI` | S2 band to use (`TCI`, `B04`, `B08`) |
| `--osm` | off | Download and rasterize OSM data |
| `--gri` | off | Download S2 GRI reference data |
| `--buffer` | `1000` | Buffer (m) used to extend the S2 GRI crop |
| `--no_buffer` | off | Disable buffer extension (writes to `S2_crop/` instead of `S2_crop_extended/`) |
| `--clc` | – | Path to a CLC raster; enables OSM gap-filling |
| `--no_data_threshold` | – | Percentage of accepted no-data in an OSM raster |
| `--rm_no_data` | off | Remove OSM rasters exceeding the no-data threshold |
| `--merge_data` | off | Write side-by-side A\|B (S2 \| OSM) images |
| `--multiprocess`, `-mp` | off | Use multiprocessing for chip extraction |
| `--topo` | off | Extract DEM/topography chips |
| `--topo_dir` | – | Path to topography data directory |
| `--highway_threshold` | `1000` | Threshold to filter S2-invisible minor roads (sampled on B04) |
| `--aoi` | – | Vector AOI path (activates AOI mode, replaces S2 input) |
| `--full_tile` | off | Tile the full S2 footprint (activates full-tile mode) |
| `--chip_size` | `2560.0` | Chip side length in meters (2560 m = 256 px @ 10 m) |
| `--osm_batch_max_deg` | `0.2` | Max bbox extent (deg) per Overpass batch (caps memory) |

---

## Data collection

**CLC**

CLC raster is available for download [here](https://land.copernicus.eu/en/products/clc-backbone/clc-backbone-2021#download)


**Topography**

To use topography with the HR prototype:
* Download DEM (Digital Elevation Model) files interpolated at 10m from [DEM4S2](https://visioterra.org/DEM4S2/)
* Convert DEM rasters to uint8 and extract patches matching OSM raster patches extents
* Use [GDAL](https://gdal.org/en/stable/programs/gdaldem.html) to convert DEM files to hillshades: both DEM or hillshades can be used


## Script roles

| File | Purpose |
|---|---|
| `dataset_creation.py` | Entry point. Parses the CLI into `HRPipelineConfig` and dispatches to the S2, full-tile, or AOI pipeline; orchestrates folder creation, chip extraction, OSM, CLC fill, topo, and AB merge. |
| `image_chips_extractor.py` | Downloads GRI chips and cuts co-registered S2 crops; tiles the full S2 footprint in full-tile mode. Empty (all-black) chips are dropped. |
| `osm_extractor.py` | Downloads OSM features for a chip/extent (single Overpass call), assigns colours, filters S2-invisible highways, and rasterizes to a 3-band RGB GeoTIFF with GDAL/OGR. |
| `clc_processing.py` | Fills OSM no-data gaps using CORINE Land Cover, then resamples to the 10 m target resolution. |
| `topography_processing.py` | Crops DEM/topography rasters to each chip's extent. |
| `geo_utils.py` | Path/stem helpers, MGRS tile parsing, AOI chip-grid generation, reference-raster creation, and raster cropping. |
| `create_train_test_split.py` | Splits an output tree into `train/` and `test/` (A/B, optionally TOPO). |
| `create_seasonal_train_test_split.py` | Builds a seasonal split with summer/winter A/B subfolders. |
| `single_class_tiles.py` | Removes (OSM, S2) pairs whose OSM raster is dominated by a single palette colour. |

---

## OSM source tags used for download

The OSM step issues a **single Overpass call** per chip/extent covering five tag families, then splits the result into roads (`highway`) and other tags for styling:

| Tag family | OSM tag |
|---|---|
| Buildings | `building=True` |
| Land use | `landuse=True` |
| Natural | `natural=True` |
| Water | `water=True` |
| Roads | `highway=True` |

Minor roads (`track`, `footway`, `path`, `residential`, `living_street`) are additionally filtered against the S2 B04 raster and dropped where they are not visible at 10 m resolution (see `--highway_threshold`).

---

## OSM styling (palette + GDAL rasterization)

Colours are assigned in Python from the fixed palette in `config.py` (`color_dict`, `highway_colors`, `natural_colors`, `landuse_colors`) and rasterized with `gdal.RasterizeLayer` at a `RASTERIZE_UPSAMPLE_FACTOR` (2×) intermediate resolution, then written as a 3-band `uint8` GeoTIFF. Features are painted in the order **natural → landuse → highway → building** (later layers overwrite earlier ones). With `--clc`, CORINE Land Cover fills remaining no-data before resampling to the 10 m `TARGET_RESOLUTION`.

This palette is intentionally coarse (a handful of semantic classes) because fine detail is not distinguishable at 10 m — in contrast to the VHR pipeline's rich per-class QGIS symbology.

---

## Output structure

For each S2 product (or AOI), the pipeline writes:

```text
<output_dir>/
└── <product_name>/                     # or <aoi_name> in AOI mode
    ├── <band>/
    │   └── S2_crop_extended/            # S2_crop/ when --no_buffer
    │       └── <chip>.tif
    ├── OSM_5m/
    │   ├── geojson/<chip>_complete.geojson
    │   ├── image/<chip>.tif             # rasterized OSM (RGB)
    │   └── image_filled/                # with --clc
    │       ├── CLC_crop/
    │       ├── 10m/
    │       └── tmp/
    ├── TOPO/                            # with --topo
    └── AB_<band>_5m/                    # with --merge_data (S2 | OSM side by side)
```

AOI mode additionally writes a `ref_rasters/` folder of blank rasters that define each chip's CRS, extent and resolution.

---

## Post-processing (train/test split)

After dataset creation, build training splits:

```bash
# Standard split of the merged A|B images (needs --merge_data)
#   -> <split>/{train,test}/*.tif, for --dataset_mode aligned
python -m GenCP_data_preparation.HR.create_train_test_split \
    --input_dir <out> --output_dir <split> --test_ratio 0.2 --seed 42 [--symlink]

# Topography split (needs --topo --topo_dir <DEM dir>)
#   -> <split>/{train,test}/{A,B,TOPO}/, for --dataset_mode dem / dem_single
python -m GenCP_data_preparation.HR.create_train_test_split --mode topography \
    --input_dir <out> --output_dir <split> \
    --subfolder_a OSM_5m/image_filled/10m \
    --subfolder_b TCI/S2_crop_extended \
    --subfolder_topo TOPO \
    --test_ratio 0.2 --seed 42 [--symlink]

# Seasonal split with summer/winter A/B subfolders
python -m GenCP_data_preparation.HR.create_seasonal_train_test_split \
    --summer_input <summer_out> --winter_input <winter_out> \
    --output_dir <split> --test_ratio 0.2 --seed 42 [--symlink]

# Drop OSM tiles dominated by a single class (dry-run first)
python -m GenCP_data_preparation.HR.single_class_tiles \
    --split_dir <split> --threshold 0.99 --dry_run
```

The standard split reads the merged pairs from `AB_TCI_5m/` (`--subfolder_ab AB_<band>_5m` for another band). The topography and seasonal splits read OSM rasters from `OSM_5m/image_filled/10m/` and S2 crops from `TCI/S2_crop_extended/` by default; override with `--subfolder_a OSM_5m/image` (no `--clc`) or `--subfolder_b <band>/S2_crop` (`--no_buffer`).

For training on the topography split, pass `<split>` as `--dataroot` with `--dataset_mode dem --input_nc 4`: the dataset reads `<split>/<phase>/` and uses `TOPO/` as topography folder. The topography split has no seasonal variant yet.

The seasonal split pairs summer and winter tiles by file name (a trailing `_summer`/`_winter` is ignored), so both seasons must cover the same area:

| Mode | Seasonal split | Notes |
|---|---|---|
| S2 (GRI) | yes | Chips keep the GRI chip names; summer and winter products must cover the same MGRS tiles. |
| Full tile | yes | Chips are named `T<MGRS>_tile_<col>_<row>` from the tile footprint, so one MGRS tile gives the same names in every season. |
| AOI | no | OSM rasters only (no S2 side), and OSM does not change with the season. Use them as inference input (`seasonal_single`). |

The seasonal split output is read as is by both seasonal training modes, `seasonal_paired_folders` (`--model pix2pix`) and `seasonal_geokey` (`--model pix2pix_stability`).

`--symlink` links instead of copying (saves disk). `single_class_tiles.py` supports `--dry_run` to preview removals and `--only {train,test}` to restrict a run. It handles topography (`A/`, `B/`, `TOPO/`) and seasonal (`A_summer/`, `B_summer/`, `A_winter/`, `B_winter/`) splits and removes a tile from every folder (merged A|B splits are not supported). Pixels within `--color_tolerance` (default 8) of a palette colour count as that colour, for lossy-compressed rasters.

---

## Notes

- OSM download requires **osmnx >= 2.0.7**; bounding boxes follow the canonical `(west, south, east, north)` order used across `GenCP_data_preparation/common/`.
- No QGIS is needed — OSM masks are produced entirely with GDAL/OGR.
- The HR and VHR pipelines keep separate OSM class→colour definitions by design (HR: Python palette; VHR: QML symbology).


### Overpass connectivity

OSM download goes through the Overpass API. `common/osm_download.py` sets a descriptive HTTP User-Agent because Overpass endpoints reject osmnx's default UA with **HTTP 406**. Behind a stricter endpoint you can override it without editing code:

```bash
export GENCP_OSM_USER_AGENT="Mozilla/5.0 (compatible; GenCP-data-prep/1.0)"
```

If downloads fail with connection errors, check network/proxy reachability to `https://overpass-api.de/api`. You can switch endpoint (a mirror, or a local Overpass instance) without editing code:

```bash
export GENCP_OVERPASS_URL="https://overpass.kumi.systems/api"   # mirror
# export GENCP_OVERPASS_URL="http://localhost:12345/api"        # local instance
```

**Rate limiting.** Full-tile mode over a whole S2 footprint issues many Overpass queries; the public `overpass-api.de` may temporarily block your IP after heavy use (connection refused). When a batched fetch hits a network/rate-limit error the pipeline stops cleanly rather than hammering the server — just **wait and rerun** (already-written chips are skipped, so it resumes). For sustained full-tile work, use a local Overpass instance via `GENCP_OVERPASS_URL`.
