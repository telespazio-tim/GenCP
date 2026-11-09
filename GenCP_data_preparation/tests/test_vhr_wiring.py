"""Verify the VHR osm_processor delegates bbox computation to common.geo.

Only the non-QGIS module (osm_processor.py) is exercised here. osm_style_processor.py
imports QGIS/PyQt5 at module load and parses CLI args, so it cannot be imported in a
plain Python environment; its one-line delegation to common.raster_io must be smoke
-tested inside a QGIS runtime separately.
"""

import importlib.util
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from GenCP_data_preparation.common import geo, osm_download

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OSM_PROCESSOR = _REPO_ROOT / "GenCP_data_preparation" / "VHR" / "osm_processor.py"


def _load_osm_processor():
    spec = importlib.util.spec_from_file_location("vhr_osm_processor", _OSM_PROCESSOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_osm_processor_imports_and_delegates(tmp_path):
    mod = _load_osm_processor()

    vpath = str(tmp_path / "v.geojson")
    gpd.GeoDataFrame(
        {"id": [1]}, geometry=[box(400000, 4999000, 401000, 5000000)], crs="EPSG:32631"
    ).to_file(vpath)

    got_bbox, got_crs = mod.compute_bbox_from_vector(vpath, buffer_meters=250)
    exp_bbox, exp_crs = geo.bbox_from_vector(vpath, buffer_meters=250)

    assert got_bbox == pytest.approx(exp_bbox, abs=1e-9)
    assert str(got_crs) == str(exp_crs)


def test_download_once_uses_canonical_tuple(monkeypatch):
    """VHR must call osmnx 2.x with a canonical (w, s, e, n) tuple, not 1.9 kwargs."""
    mod = _load_osm_processor()

    captured = {}

    def fake(bbox=None, tags=None):
        captured["bbox"] = bbox
        return gpd.GeoDataFrame({"geometry": [Point(2.5, 48.5)]}, crs="EPSG:4326")

    monkeypatch.setattr(osm_download.ox, "features_from_bbox", fake)

    bbox = (2.0, 48.0, 3.0, 49.0)  # canonical (w, s, e, n)
    mod._download_once(bbox, {"building": True})
    assert captured["bbox"] == bbox


def test_split_and_tags_delegate():
    mod = _load_osm_processor()
    assert mod.split_bbox_wgs84((0.0, 0.0, 10.0, 10.0), 2) == osm_download.split_bbox((0.0, 0.0, 10.0, 10.0), 2)
    specs = {"a": {"tags": {"building": True}}, "b": {"tags": {"landuse": True}}}
    assert mod.build_combined_tags(specs) == {"building": True, "landuse": True}
