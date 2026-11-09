"""Tests for GenCP_data_preparation.common.geo.

Run from the repository root:  python -m pytest GenCP_data_preparation/tests
"""

import geopandas as gpd
import numpy as np
import pyproj
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from shapely.geometry import box

from GenCP_data_preparation.common import geo


# ---------------------------------------------------------------------------
# UTM estimation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "lon, lat, epsg",
    [
        (2.35, 48.85, 32631),    # Paris -> UTM 31N
        (-0.13, 51.5, 32630),    # London -> UTM 30N
        (-58.4, -34.6, 32721),   # Buenos Aires -> UTM 21S
        (139.7, 35.7, 32654),    # Tokyo -> UTM 54N
    ],
)
def test_estimate_utm_crs(lon, lat, epsg):
    assert geo.estimate_utm_crs(lon, lat).to_epsg() == epsg


def test_hr_wrapper_matches_shared():
    """HR's _estimate_utm_crs must delegate to the shared implementation."""
    from GenCP_data_preparation.HR.geo_utils import _estimate_utm_crs

    assert _estimate_utm_crs(2.35, 48.85).to_epsg() == geo.estimate_utm_crs(2.35, 48.85).to_epsg()


# ---------------------------------------------------------------------------
# Ordering helpers
# ---------------------------------------------------------------------------
def test_order_helpers_roundtrip():
    wsen = (1.0, 2.0, 3.0, 4.0)  # west, south, east, north
    assert geo.wsen_to_nsew(wsen) == (4.0, 2.0, 3.0, 1.0)
    assert geo.nsew_to_wsen(geo.wsen_to_nsew(wsen)) == wsen


# ---------------------------------------------------------------------------
# reproject_bounds
# ---------------------------------------------------------------------------
def test_reproject_bounds_identity():
    b = (2.0, 48.0, 3.0, 49.0)
    out = geo.reproject_bounds(b, "EPSG:4326", "EPSG:4326")
    assert out == pytest.approx(b, abs=1e-9)


# ---------------------------------------------------------------------------
# bbox_from_raster
# ---------------------------------------------------------------------------
def _make_raster(path, crs="EPSG:32631"):
    # 100x100 px @ 10 m, origin at (400000 E, 5000000 N) in UTM 31N
    transform = from_origin(400000, 5000000, 10, 10)
    profile = dict(
        driver="GTiff", height=100, width=100, count=1,
        dtype="uint8", crs=crs, transform=transform,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.zeros((1, 100, 100), "uint8"))
    # native bounds: (400000, 4999000, 401000, 5000000)
    return (400000.0, 4999000.0, 401000.0, 5000000.0)


def test_bbox_from_raster(tmp_path):
    rpath = str(tmp_path / "r.tif")
    native = _make_raster(rpath)

    (w, s, e, n), crs = geo.bbox_from_raster(rpath)

    assert pyproj.CRS(crs).to_epsg() == 32631
    assert w < e and s < n                      # canonical order
    # UTM 31N easting 400000 is ~100 km west of the 3degE central meridian.
    assert 1.0 < w < 3.0 and 45.0 < s < 46.0

    # matches a direct densified transform of the native bounds
    expected = transform_bounds("EPSG:32631", "EPSG:4326", *native, densify_pts=21)
    assert (w, s, e, n) == pytest.approx(expected, abs=1e-9)


def test_bbox_from_raster_buffer_expands(tmp_path):
    rpath = str(tmp_path / "r.tif")
    _make_raster(rpath)

    (w0, s0, e0, n0), _ = geo.bbox_from_raster(rpath, buffer_meters=0)
    (w1, s1, e1, n1), _ = geo.bbox_from_raster(rpath, buffer_meters=500)

    assert w1 < w0 and s1 < s0 and e1 > e0 and n1 > n0


def test_bbox_from_raster_no_crs_raises(tmp_path):
    rpath = str(tmp_path / "nocrs.tif")
    with rasterio.open(
        rpath, "w", driver="GTiff", height=10, width=10, count=1,
        dtype="uint8", transform=from_origin(0, 10, 1, 1),
    ) as dst:
        dst.write(np.zeros((1, 10, 10), "uint8"))
    with pytest.raises(ValueError):
        geo.bbox_from_raster(rpath)


# ---------------------------------------------------------------------------
# bbox_from_vector
# ---------------------------------------------------------------------------
def _write_vector(path, geom, crs):
    gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs=crs).to_file(path)


def test_bbox_from_vector_wgs84(tmp_path):
    vpath = str(tmp_path / "v.geojson")
    _write_vector(vpath, box(2.0, 48.0, 3.0, 49.0), "EPSG:4326")

    (w, s, e, n), crs = geo.bbox_from_vector(vpath)

    assert pyproj.CRS(crs).to_epsg() == 4326
    assert (w, s, e, n) == pytest.approx((2.0, 48.0, 3.0, 49.0), abs=1e-6)


def test_bbox_from_vector_projected_and_buffer(tmp_path):
    vpath = str(tmp_path / "v.geojson")
    _write_vector(vpath, box(400000, 4999000, 401000, 5000000), "EPSG:32631")

    (w0, s0, e0, n0), crs = geo.bbox_from_vector(vpath)
    assert pyproj.CRS(crs).to_epsg() == 32631
    assert w0 < e0 and s0 < n0

    (w1, s1, e1, n1), _ = geo.bbox_from_vector(vpath, buffer_meters=1000)
    assert w1 < w0 and s1 < s0 and e1 > e0 and n1 > n0


def test_bbox_from_vector_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        geo.bbox_from_vector("/no/such/file.geojson")
