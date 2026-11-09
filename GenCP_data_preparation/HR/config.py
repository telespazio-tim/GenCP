"""Unified configuration for the HR data preparation pipeline.

Contains color dictionaries, named constants, and the HRPipelineConfig dataclass.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..common.config import BaseConfig


# ---------------------------------------------------------------------------
# Color dictionaries (module-level for backward compatibility)
# ---------------------------------------------------------------------------

color_dict = {
    "water": "#80ccff",
    "light_green": "#85e085",
    "forest_green": "#009933",
    "sand": "#ffffcc",
    "light_purple": "#eeccff",
    "light_gray": "#e6e6e6",
    "gray": "#cccccc",
    "rock": "#d9b38c",
    "red_road": "#ff5050",
    "orange_road": "#ff944d",
    "medium_orange_road": "#ffa64d",
    "light_orange_road": "#ffbf80",
    "residential_road": "#fff3e6",
    "tertiary_road": "#ffd9b3",
    "foot_path": "#ecd9c6",
    "track": "#dfbf9f",
    "salt_pond": "#eeccff",
    "unclassified_road": "#ffe6cc",
    "snow": "white",
    "black": "black",
    "no_vegetation": "#C3BA8D",
}

highway_colors = {
    "motorway": "red_road",
    "trunk": "orange_road",
    "primary": "medium_orange_road",
    "secondary": "light_orange_road",
    "residential": "residential_road",
    "living_street": "residential_road",
    "service": "residential_road",
    "tertiary": "tertiary_road",
    "unclassified": "unclassified_road",
    "road": "unclassified_road",
    "track": "track",
    "footway": "foot_path",
    "path": "foot_path",
}

natural_colors = {
    "wood": "forest_green",
    "fell": "light_green",
    "grassland": "light_green",
    "scrub": "light_green",
    "shrubbery": "light_green",
    "tundra": "light_green",
    "valley": "light_green",
    "hill": "light_green",
    "heath": "light_green",
    "bay": "water",
    "water": "water",
    "wetland": "light_green",
    "strait": "water",
    "reef": "sand",
    "shoal": "sand",
    "beach": "sand",
    "sand": "sand",
    "bare_rock": "rock",
    "blockfield": "rock",
    "scree": "rock",
}

landuse_colors = {
    "paddy": "light_green",
    "greenfield": "light_green",
    "vineyard": "light_green",
    "cemetery": "light_green",
    "grass": "light_green",
    "port": "water",
    "basin": "water",
    "aquaculture": "water",
    "forest": "forest_green",
    "greenhouse_horticulture": "light_green",
    "farmland": "light_green",
    "meadow": "light_green",
    "orchard": "light_green",
    "salt_pond": "salt_pond",
    "commercial": "light_gray",
    "education": "light_gray",
    "retail": "light_gray",
    "industrial": "gray",
    "construction": "gray",
    "residential": "light_purple",
}

CLC_color_mapping = {
    "1": "gray",
    "2": "forest_green",
    "3": "forest_green",
    "4": "forest_green",
    "5": "light_green",
    "6": "light_green",
    "7": "light_green",
    "8": "light_green",
    "9": "no_vegetation",
    "10": "water",
    "11": "snow",
    "253": "water",
    "254": "water",
    "0": "black",
}


# ---------------------------------------------------------------------------
# Named constants (replace hardcoded magic numbers across the pipeline)
# ---------------------------------------------------------------------------

BBOX_BUFFER_METERS: int = 100
"""Buffer in meters added to bounding box when extracting OSM data."""

DEFAULT_HIGHWAY_WIDTH: float = 1.0
"""Default buffer width in meters for highway line features."""

RASTERIZE_UPSAMPLE_FACTOR: int = 2
"""Factor by which rasterization resolution is improved relative to input."""

TARGET_RESOLUTION: float = 10.0
"""Target output resolution in meters for the final OSM raster."""

QUICKLOOK_SRC_MAX: int = 2500
"""Default maximum source value for quicklook radiometric scaling."""

QUICKLOOK_SRC_MAX_SWIR: int = 4000
"""Maximum source value for SWIR band quicklooks (B12, B11, B8A)."""

QUICKLOOK_QUANTIFICATION: float = 10000.0
"""Quantification value for Sentinel-2 reflectance scaling."""

QUICKLOOK_SITE_OVERRIDES: dict = {
    "34TCR": {"default": 4000, "swir": 10000},
}
"""Per-MGRS-tile overrides for quicklook scaling parameters."""

GRI_BASE_URL: str = (
    "https://s3.eu-central-2.wasabisys.com/s2-mpc/data/S2_GRI/GCP_L1C"
)
"""Base URL for downloading S2 GRI (Ground Reference Image) chips."""

DEFAULT_CHIP_SIZE_METERS: float = 2560.0
"""Default chip side length in meters (256 pixels at 10 m resolution)."""


# ---------------------------------------------------------------------------
# Pipeline configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class HRPipelineConfig(BaseConfig):
    """Central configuration for the HR data preparation pipeline.

    ``input_dir`` / ``output_dir`` (and their Path coercion) are inherited from
    :class:`GenCP_data_preparation.common.config.BaseConfig`.
    """

    # Paths (input_dir / output_dir inherited from BaseConfig)
    clc_path: Optional[Path] = None
    topo_dir: Optional[Path] = None
    aoi_path: Optional[Path] = None

    # Band selection
    band: str = "TCI"

    # Processing flags
    download_osm: bool = False
    download_gri: bool = False
    apply_buffer: bool = True
    buffer_size: int = 1000
    merge_data: bool = False
    use_multiprocess: bool = False
    extract_topo: bool = False

    # AOI / full-tile mode
    chip_size: float = DEFAULT_CHIP_SIZE_METERS
    full_tile: bool = False

    # Thresholds
    highway_threshold: int = 1000
    no_data_threshold: Optional[int] = None
    rm_no_data: bool = False

    # OSM batching
    osm_batch_max_deg: float = 0.2

    @property
    def is_aoi_mode(self) -> bool:
        """True when the pipeline runs from a vector AOI instead of S2 products."""
        return self.aoi_path is not None

    @property
    def is_full_tile_mode(self) -> bool:
        """True when the pipeline tiles each S2 image's full footprint into chips."""
        return self.full_tile and self.aoi_path is None

    @property
    def s2_crop_folder(self) -> str:
        return "S2_crop_extended" if self.apply_buffer else "S2_crop"

    def __post_init__(self):
        super().__post_init__()  # coerces input_dir / output_dir to Path
        if self.clc_path is not None:
            self.clc_path = Path(self.clc_path)
        if self.topo_dir is not None:
            self.topo_dir = Path(self.topo_dir)
        if self.aoi_path is not None:
            self.aoi_path = Path(self.aoi_path)

        if not self.is_aoi_mode and self.band not in {"B04", "B08", "TCI"}:
            raise ValueError(
                f"Invalid band '{self.band}'. Must be one of: B04, B08, TCI"
            )

    @classmethod
    def from_args(cls, args) -> "HRPipelineConfig":
        """Create config from argparse namespace."""
        return cls(
            input_dir=Path(args.input_dir) if args.input_dir else Path("."),
            output_dir=Path(args.output_dir) if args.output_dir else Path("."),
            clc_path=Path(args.clc) if args.clc else None,
            topo_dir=Path(args.topo_dir) if args.topo_dir else None,
            aoi_path=Path(args.aoi) if getattr(args, "aoi", None) else None,
            band=args.band,
            download_osm=args.osm,
            download_gri=args.gri,
            apply_buffer=not args.no_buffer,
            buffer_size=args.buffer,
            merge_data=args.merge_data,
            use_multiprocess=args.multiprocess,
            extract_topo=args.topo,
            chip_size=getattr(args, "chip_size", DEFAULT_CHIP_SIZE_METERS),
            full_tile=getattr(args, "full_tile", False),
            highway_threshold=args.highway_threshold,
            no_data_threshold=args.no_data_threshold,
            rm_no_data=args.rm_no_data,
            osm_batch_max_deg=getattr(args, "osm_batch_max_deg", 0.2),
        )
