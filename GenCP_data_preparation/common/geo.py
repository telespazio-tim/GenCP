"""Shared geospatial helpers: bounding boxes, UTM estimation, bounds reprojection.

Bounding-box convention
-----------------------
Every function in this module that returns a bounding box uses the **canonical
order** ``(west, south, east, north)`` — i.e. ``(min_lon, min_lat, max_lon,
max_lat)``. This matches ``shapely``/``geopandas`` ``total_bounds`` and
``rasterio`` bounds, and is the single order used internally by the harmonized
pipelines.

Callers that need a different order (e.g. the ``(north, south, east, west)``
tuple historically consumed by the HR osmnx code path) must convert explicitly
with :func:`wsen_to_nsew`. Keeping the conversion at the call site — rather than
having every helper return a different order — is what removes the silent
ordering mismatches that previously existed between HR and VHR.
"""

import logging
from pathlib import Path
from typing import Tuple, Union

import geopandas as gpd
import pyproj
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from shapely.geometry import box

logger = logging.getLogger(__name__)

# A bounding box in canonical (west, south, east, north) order.
BBox = Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Ordering helpers
# ---------------------------------------------------------------------------
def wsen_to_nsew(bbox: BBox) -> Tuple[float, float, float, float]:
    """Convert a canonical ``(west, south, east, north)`` bbox to ``(n, s, e, w)``.

    This is the order the legacy HR osmnx code path expects.
    """
    west, south, east, north = bbox
    return (north, south, east, west)


def nsew_to_wsen(bbox: Tuple[float, float, float, float]) -> BBox:
    """Convert a ``(north, south, east, west)`` bbox to canonical ``(w, s, e, n)``."""
    north, south, east, west = bbox
    return (west, south, east, north)


# ---------------------------------------------------------------------------
# UTM estimation
# ---------------------------------------------------------------------------
def estimate_utm_crs(lon: float, lat: float) -> pyproj.CRS:
    """Estimate an appropriate UTM CRS for a given lon/lat point."""
    zone_number = int((lon + 180) / 6) + 1
    hemisphere = "north" if lat >= 0 else "south"
    epsg = 32600 + zone_number if hemisphere == "north" else 32700 + zone_number
    return pyproj.CRS.from_epsg(epsg)


# ---------------------------------------------------------------------------
# Bounds reprojection
# ---------------------------------------------------------------------------
def reproject_bounds(
    bounds: BBox,
    src_crs,
    dst_crs="EPSG:4326",
    densify_pts: int = 21,
) -> BBox:
    """Reproject a ``(west, south, east, north)`` bbox to ``dst_crs``.

    Uses ``rasterio.warp.transform_bounds`` with edge densification so the
    reprojected envelope correctly encloses the source extent (rather than only
    transforming the two corner points).
    """
    west, south, east, north = bounds
    return transform_bounds(
        src_crs, dst_crs, west, south, east, north, densify_pts=densify_pts
    )


# ---------------------------------------------------------------------------
# Bounding box from raster / vector
# ---------------------------------------------------------------------------
def bbox_from_raster(
    input_path: Union[str, Path],
    buffer_meters: float = 0,
    densify_pts: int = 21,
) -> Tuple[BBox, object]:
    """Return a raster's extent as an EPSG:4326 bbox plus its native CRS.

    Args:
        input_path: Path to a georeferenced raster.
        buffer_meters: Optional buffer, applied in the raster's native CRS
            (assumed metric) before reprojection.
        densify_pts: Edge densification for the reprojection.

    Returns:
        ``((west, south, east, north), raster_crs)`` with the bbox in EPSG:4326.

    Raises:
        ValueError: If the raster has no CRS.
    """
    with rasterio.open(input_path) as ds:
        if ds.crs is None:
            raise ValueError(f"Raster file is missing CRS information: {input_path}")

        raster_crs = ds.crs
        b = ds.bounds
        left = b.left - buffer_meters
        bottom = b.bottom - buffer_meters
        right = b.right + buffer_meters
        top = b.top + buffer_meters

    bbox_4326 = transform_bounds(
        raster_crs,
        CRS.from_epsg(4326),
        left,
        bottom,
        right,
        top,
        densify_pts=densify_pts,
    )
    return bbox_4326, raster_crs


def bbox_from_vector(
    input_path: Union[str, Path],
    buffer_meters: float = 0,
) -> Tuple[BBox, object]:
    """Return a vector's extent as an EPSG:4326 bbox plus its native CRS.

    The native bounding box is reprojected to EPSG:4326 (matching the behaviour
    the VHR pipeline has always used). When ``buffer_meters`` is positive, the
    buffer is applied in an estimated UTM CRS before reprojection.

    Args:
        input_path: Path to a GeoJSON / shapefile / other vector source.
        buffer_meters: Optional buffer in meters, applied in an estimated UTM CRS.

    Returns:
        ``((west, south, east, north), native_crs)`` with the bbox in EPSG:4326.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the vector is empty or has no CRS.
    """
    p = Path(input_path)
    if not p.is_file():
        raise FileNotFoundError(f"Vector file not found: {p}")

    gdf = gpd.read_file(p)
    if gdf.empty:
        raise ValueError(f"Vector file is empty: {p}")
    if gdf.crs is None:
        raise ValueError(f"Vector file is missing CRS information: {p}")

    native_crs = gdf.crs

    if buffer_meters and buffer_meters > 0:
        work_crs = gdf.estimate_utm_crs()
        minx, miny, maxx, maxy = gdf.to_crs(work_crs).total_bounds
        geom = box(
            minx - buffer_meters,
            miny - buffer_meters,
            maxx + buffer_meters,
            maxy + buffer_meters,
        )
    else:
        work_crs = native_crs
        minx, miny, maxx, maxy = gdf.total_bounds
        geom = box(minx, miny, maxx, maxy)

    west, south, east, north = (
        gpd.GeoSeries([geom], crs=work_crs).to_crs("EPSG:4326").total_bounds
    )
    return (west, south, east, north), native_crs
