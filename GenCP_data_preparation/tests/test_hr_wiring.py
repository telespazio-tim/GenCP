"""Verify HR's chip extractor uses the shared empty-tile filter, and that the
bbox order / OSM download were reconciled onto the canonical (w, s, e, n) order.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import requests
from osgeo import ogr, osr
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

from GenCP_data_preparation.common import geo, osm_download
from GenCP_data_preparation.common.geo import bbox_from_raster
from GenCP_data_preparation.HR import dataset_creation, osm_extractor
from GenCP_data_preparation.HR.config import BBOX_BUFFER_METERS
from GenCP_data_preparation.HR.image_chips_extractor import _crop_chip_with_window


def _make_band(path, fill):
    transform = from_origin(400000, 5000000, 10, 10)  # 10 m px, UTM-like
    profile = dict(
        driver="GTiff", height=256, width=256, count=1,
        dtype="uint8", crs="EPSG:32631", transform=transform,
    )
    with rasterio.open(path, "w", **profile) as dst:
        arr = np.full((1, 256, 256), fill, "uint8")
        dst.write(arr)
    # chip covering the whole raster
    return {
        "bounds": (400000.0, 5000000.0 - 2560.0, 400000.0 + 2560.0, 5000000.0),
        "width": 256,
        "height": 256,
    }


def test_all_black_chip_rejected(tmp_path):
    bpath = str(tmp_path / "b.tif")
    chip = _make_band(bpath, fill=0)
    out = str(tmp_path / "chip.tif")
    with rasterio.open(bpath) as src:
        assert _crop_chip_with_window(src, chip, out) is False
    assert not (tmp_path / "chip.tif").exists()


def test_nonempty_chip_written(tmp_path):
    bpath = str(tmp_path / "b.tif")
    chip = _make_band(bpath, fill=42)
    out = str(tmp_path / "chip.tif")
    with rasterio.open(bpath) as src:
        assert _crop_chip_with_window(src, chip, out) is True
    assert (tmp_path / "chip.tif").exists()


# ---------------------------------------------------------------------------
# bbox-order reconciliation (Phase 3): extract_bbox_from_image must now return
# canonical (west, south, east, north), matching osmnx 2.x — the previous
# (north, south, west, east) order was a transposition bug.
# ---------------------------------------------------------------------------
def test_extract_bbox_from_image_is_canonical(tmp_path):
    rpath = str(tmp_path / "b.tif")
    transform = from_origin(400000, 5000000, 10, 10)
    with rasterio.open(
        rpath, "w", driver="GTiff", height=256, width=256, count=1,
        dtype="uint8", crs="EPSG:32631", transform=transform,
    ) as dst:
        dst.write(np.zeros((1, 256, 256), "uint8"))

    bbox = osm_extractor.extract_bbox_from_image(rpath)
    w, s, e, n = bbox
    assert w < e and s < n                                    # canonical order
    expected, _ = bbox_from_raster(rpath, buffer_meters=BBOX_BUFFER_METERS)
    assert bbox == pytest.approx(expected)                    # delegates to shared helper


def test_extract_osm_data_forwards_canonical_bbox(monkeypatch):
    """HR's OSM download must hand osmnx the canonical bbox unchanged."""
    captured = {}

    def fake(bbox=None, tags=None):
        captured["bbox"] = bbox
        return gpd.GeoDataFrame(
            {"highway": [None, "residential"], "building": ["yes", None]},
            geometry=[Point(2.5, 48.5), LineString([(2.5, 48.5), (2.6, 48.6)])],
            crs="EPSG:4326",
        )

    monkeypatch.setattr(osm_download.ox, "features_from_bbox", fake)

    bbox = (2.0, 48.0, 3.0, 49.0)  # canonical (w, s, e, n)
    tags_data, roads_data = osm_extractor.extract_osm_data(bbox, "chip")

    assert captured["bbox"] == bbox
    assert roads_data["features"] and tags_data["features"]


# ---------------------------------------------------------------------------
# Batched OSM fetch: network errors must hard-stop (not fan out to per-chip),
# while non-network errors keep the per-chip fallback.
# ---------------------------------------------------------------------------
def test_batched_fetch_network_error_aborts_without_per_chip(monkeypatch, tmp_path):
    monkeypatch.setattr(osm_extractor, "extract_bbox_from_image", lambda p: (2.0, 48.0, 3.0, 49.0))

    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("Connection refused")

    monkeypatch.setattr(osm_extractor, "extract_osm_data_for_extent", boom)

    called = {"per_chip": 0}
    monkeypatch.setattr(osm_extractor, "create_osm_raster",
                        lambda *a, **k: called.__setitem__("per_chip", called["per_chip"] + 1))

    with pytest.raises(RuntimeError, match="Overpass request failed"):
        osm_extractor.create_osm_rasters_for_chips(
            ["chipA.tif"], str(tmp_path), {}, {}, {}, {}, highway_threshold=1000,
        )
    assert called["per_chip"] == 0   # network error must NOT trigger per-chip fan-out


def test_batched_fetch_overpass_502_aborts_without_per_chip(monkeypatch, tmp_path):
    from osmnx._errors import ResponseStatusCodeError
    monkeypatch.setattr(osm_extractor, "extract_bbox_from_image", lambda p: (2.0, 48.0, 3.0, 49.0))

    def boom(*a, **k):
        raise ResponseStatusCodeError("'overpass-api.de' responded: 502 Bad Gateway")

    monkeypatch.setattr(osm_extractor, "extract_osm_data_for_extent", boom)

    called = {"per_chip": 0}
    monkeypatch.setattr(osm_extractor, "create_osm_raster",
                        lambda *a, **k: called.__setitem__("per_chip", called["per_chip"] + 1))

    with pytest.raises(RuntimeError, match="Overpass request failed"):
        osm_extractor.create_osm_rasters_for_chips(
            ["chipA.tif", "chipB.tif"], str(tmp_path), {}, {}, {}, {}, highway_threshold=1000,
        )
    assert called["per_chip"] == 0   # overloaded server must NOT trigger per-chip fan-out


def test_batched_fetch_other_error_falls_back_to_per_chip(monkeypatch, tmp_path):
    monkeypatch.setattr(osm_extractor, "extract_bbox_from_image", lambda p: (2.0, 48.0, 3.0, 49.0))

    def boom(*a, **k):
        raise ValueError("some non-network parsing error")

    monkeypatch.setattr(osm_extractor, "extract_osm_data_for_extent", boom)

    called = {"per_chip": 0}
    monkeypatch.setattr(osm_extractor, "create_osm_raster",
                        lambda *a, **k: called.__setitem__("per_chip", called["per_chip"] + 1))

    # No raise: data error keeps the original per-chip fallback behavior.
    osm_extractor.create_osm_rasters_for_chips(
        ["chipA.tif"], str(tmp_path), {}, {}, {}, {}, highway_threshold=1000,
    )
    assert called["per_chip"] == 1


# ---------------------------------------------------------------------------
# Review fix #1: OGR reprojection must use traditional (lon, lat) axis order so
# GeoJSON (lon, lat) OSM geometries land at the correct UTM coordinates under
# GDAL >= 3 (authority order would swap them).
# ---------------------------------------------------------------------------
def test_reproject_layer_uses_traditional_axis_order():
    # Source layer in EPSG:4326 built the way _features_to_memory_ogr_layer does
    # (ImportFromEPSG, no explicit axis strategy), one point in (lon, lat).
    src_srs = osr.SpatialReference()
    src_srs.ImportFromEPSG(4326)
    mem = ogr.GetDriverByName("Memory").CreateDataSource("")
    layer = mem.CreateLayer("src", srs=src_srs)
    feat = ogr.Feature(layer.GetLayerDefn())
    feat.SetGeometry(ogr.CreateGeometryFromWkt("POINT (2.35 48.85)"))  # Paris (lon, lat)
    layer.CreateFeature(feat)

    img_srs = osr.SpatialReference()
    img_srs.ImportFromEPSG(32631)  # UTM 31N

    out_layer, _ds = osm_extractor._reproject_layer_if_needed(layer, img_srs)
    out_layer.ResetReading()
    feat_out = out_layer.GetNextFeature()          # hold the feature reference
    g = feat_out.GetGeometryRef().Clone()          # clone so it outlives the feature
    # Correct UTM 31N for Paris ~ (452000, 5411000); swapped axes give garbage.
    assert 440000 < g.GetX() < 470000
    assert 5.40e6 < g.GetY() < 5.42e6


# ---------------------------------------------------------------------------
# Review fix #2: AOI mode must not use the absolute `from config import ...`
# (it raised ModuleNotFoundError). TARGET_RESOLUTION is now a module import.
# ---------------------------------------------------------------------------
def test_dataset_creation_has_target_resolution_at_module_level():
    assert hasattr(dataset_creation, "TARGET_RESOLUTION")


def test_no_absolute_config_import_in_dataset_creation():
    import inspect
    src = inspect.getsource(dataset_creation)
    assert "from config import" not in src  # must be the package-relative .config


# ---------------------------------------------------------------------------
# Review fix #3: tiles with OSM features but no 'highway' column must not crash
# on osmnx's (element_type, osmid) MultiIndex.
# ---------------------------------------------------------------------------
def test_extract_osm_data_no_highway_column(monkeypatch):
    idx = pd.MultiIndex.from_tuples([("way", 1), ("way", 2)], names=["element_type", "osmid"])
    no_road_gdf = gpd.GeoDataFrame(
        {"building": ["yes", "house"]},   # note: no 'highway' column at all
        geometry=[Point(2.5, 48.5), Point(2.6, 48.6)],
        index=idx, crs="EPSG:4326",
    )
    monkeypatch.setattr(osm_download.ox, "features_from_bbox", lambda bbox=None, tags=None: no_road_gdf)

    tags_data, roads_data = osm_extractor.extract_osm_data((2.0, 48.0, 3.0, 49.0), "chip")
    assert roads_data["features"] == []          # no roads, no crash
    assert len(tags_data["features"]) == 2       # buildings preserved


# ---------------------------------------------------------------------------
# Overpass query restricted to the coloured tag values
# ---------------------------------------------------------------------------
def test_build_osm_query_tags_defaults_to_config_colors():
    from GenCP_data_preparation.HR import config as hr_config

    tags = osm_extractor.build_osm_query_tags()
    assert tags["building"] is True
    assert tags["highway"] == sorted(hr_config.highway_colors)
    assert tags["natural"] == sorted(hr_config.natural_colors)
    assert tags["landuse"] == sorted(hr_config.landuse_colors)
    assert "water" not in tags   # never read by the rasterizer


def test_build_osm_query_tags_skips_empty_mappings():
    assert osm_extractor.build_osm_query_tags({}, {"wood": "forest_green"}, {}) == {
        "building": True, "natural": ["wood"],
    }


def test_create_osm_rasters_for_chips_sends_colored_tags(monkeypatch, tmp_path):
    monkeypatch.setattr(osm_extractor, "extract_bbox_from_image", lambda p: (2.0, 48.0, 2.01, 48.01))
    captured = {}

    def fake_extent(bbox, name, tags=None):
        captured["tags"] = tags
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")

    monkeypatch.setattr(osm_extractor, "extract_osm_data_for_extent", fake_extent)
    monkeypatch.setattr(osm_extractor, "_process_chip_from_shared_gdf", lambda *a, **k: None)

    osm_extractor.create_osm_rasters_for_chips(
        ["chipA.tif"], str(tmp_path), {"motorway": "red_road"}, {"wood": "forest_green"},
        {"forest": "forest_green"}, {}, highway_threshold=1000,
    )
    assert captured["tags"] == {
        "building": True, "highway": ["motorway"], "natural": ["wood"], "landuse": ["forest"],
    }
