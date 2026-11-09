"""Tests for GenCP_data_preparation.common.osm_download.

The live Overpass call is never made; osmnx is monkeypatched so we can assert the
canonical bbox is forwarded correctly (osmnx 2.x tuple API).
"""

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from GenCP_data_preparation.common import osm_download as osm


# ---------------------------------------------------------------------------
# split_bbox
# ---------------------------------------------------------------------------
def test_split_bbox_quadrants():
    parts = osm.split_bbox((0.0, 0.0, 10.0, 20.0), n=2)
    assert len(parts) == 4
    assert set(parts) == {
        (0.0, 0.0, 5.0, 10.0),
        (0.0, 10.0, 5.0, 20.0),
        (5.0, 0.0, 10.0, 10.0),
        (5.0, 10.0, 10.0, 20.0),
    }


def test_split_bbox_union_equals_input():
    bbox = (2.0, 40.0, 6.0, 44.0)
    parts = osm.split_bbox(bbox, n=3)
    assert len(parts) == 9
    ws = min(p[0] for p in parts); ss = min(p[1] for p in parts)
    es = max(p[2] for p in parts); ns = max(p[3] for p in parts)
    assert (ws, ss, es, ns) == pytest.approx(bbox)


# ---------------------------------------------------------------------------
# combine_tags
# ---------------------------------------------------------------------------
def test_combine_tags_distinct_keys():
    assert osm.combine_tags([{"building": True}, {"landuse": True}]) == {
        "building": True, "landuse": True,
    }


def test_combine_tags_true_dominates():
    assert osm.combine_tags([{"amenity": "cafe"}, {"amenity": True}]) == {"amenity": True}


def test_combine_tags_unions_values():
    assert osm.combine_tags([{"amenity": "cafe"}, {"amenity": "bar"}]) == {
        "amenity": ["bar", "cafe"],
    }


# ---------------------------------------------------------------------------
# download_features (osmnx monkeypatched)
# ---------------------------------------------------------------------------
def test_download_features_forwards_canonical_bbox(monkeypatch):
    captured = {}

    def fake_features_from_bbox(bbox=None, tags=None):
        captured["bbox"] = bbox
        captured["tags"] = tags
        return gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:4326")

    monkeypatch.setattr(osm.ox, "features_from_bbox", fake_features_from_bbox)

    bbox = (2.0, 48.0, 3.0, 49.0)  # canonical (w, s, e, n)
    out = osm.download_features(bbox, {"building": True})

    assert captured["bbox"] == bbox
    assert captured["tags"] == {"building": True}
    assert len(out) == 1


# ---------------------------------------------------------------------------
# configure_osm_client (User-Agent — avoids Overpass HTTP 406)
# ---------------------------------------------------------------------------
def test_import_sets_non_default_user_agent():
    # Importing the module must have moved osmnx off its default UA (which
    # Overpass rejects with 406).
    assert osm.ox.settings.http_user_agent == osm.DEFAULT_USER_AGENT
    assert "OSMnx" not in osm.ox.settings.http_user_agent


def test_configure_osm_client_explicit_arg():
    try:
        applied = osm.configure_osm_client("MyUA/9.9")
        assert applied == "MyUA/9.9"
        assert osm.ox.settings.http_user_agent == "MyUA/9.9"
        assert osm.ox.settings.http_referer == "MyUA/9.9"
    finally:
        osm.configure_osm_client()  # restore default for other tests


def test_configure_osm_client_env_override(monkeypatch):
    monkeypatch.setenv(osm.USER_AGENT_ENV_VAR, "EnvUA/1.0")
    try:
        assert osm.configure_osm_client() == "EnvUA/1.0"
        assert osm.ox.settings.http_user_agent == "EnvUA/1.0"
    finally:
        monkeypatch.delenv(osm.USER_AGENT_ENV_VAR, raising=False)
        osm.configure_osm_client()


def test_configure_osm_client_arg_beats_env(monkeypatch):
    monkeypatch.setenv(osm.USER_AGENT_ENV_VAR, "EnvUA/1.0")
    try:
        assert osm.configure_osm_client("ArgUA/2.0") == "ArgUA/2.0"
    finally:
        monkeypatch.delenv(osm.USER_AGENT_ENV_VAR, raising=False)
        osm.configure_osm_client()


def test_configure_overpass_url_from_env(monkeypatch):
    original = osm.ox.settings.overpass_url
    monkeypatch.setenv(osm.OVERPASS_URL_ENV_VAR, "http://localhost:12345/api")
    try:
        osm.configure_osm_client()
        assert osm.ox.settings.overpass_url == "http://localhost:12345/api"
    finally:
        monkeypatch.delenv(osm.OVERPASS_URL_ENV_VAR, raising=False)
        osm.ox.settings.overpass_url = original


def test_configure_overpass_url_unset_leaves_default(monkeypatch):
    monkeypatch.delenv(osm.OVERPASS_URL_ENV_VAR, raising=False)
    original = osm.ox.settings.overpass_url
    osm.configure_osm_client()  # no url arg, no env -> unchanged
    assert osm.ox.settings.overpass_url == original


# ---------------------------------------------------------------------------
# Retry with backoff on transient Overpass failures
# ---------------------------------------------------------------------------
def _overpass_error(status):
    from osmnx._errors import ResponseStatusCodeError
    return ResponseStatusCodeError(f"'overpass-api.de' responded: {status} Some Reason")


def test_is_transient_error_classification():
    import requests
    assert osm.is_transient_error(_overpass_error(502))
    assert osm.is_transient_error(_overpass_error(429))
    assert osm.is_transient_error(requests.exceptions.ConnectionError("x"))
    assert osm.is_transient_error(requests.exceptions.ReadTimeout("x"))
    assert not osm.is_transient_error(_overpass_error(400))
    assert not osm.is_transient_error(ValueError("x"))


def test_download_features_retries_transient_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky(bbox=None, tags=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _overpass_error(502)
        return "gdf"

    sleeps = []
    monkeypatch.setattr(osm.ox, "features_from_bbox", flaky)
    monkeypatch.setattr(osm.time, "sleep", sleeps.append)
    assert osm.download_features((0, 0, 1, 1), {"building": True}, max_retries=5, backoff_base=10) == "gdf"
    assert calls["n"] == 3
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]   # exponential backoff


def test_download_features_gives_up_after_max_retries(monkeypatch):
    from osmnx._errors import ResponseStatusCodeError

    def always_502(bbox=None, tags=None):
        raise _overpass_error(502)

    monkeypatch.setattr(osm.ox, "features_from_bbox", always_502)
    monkeypatch.setattr(osm.time, "sleep", lambda s: None)
    with pytest.raises(ResponseStatusCodeError):
        osm.download_features((0, 0, 1, 1), {"building": True}, max_retries=2, backoff_base=1)


def test_download_features_does_not_retry_non_transient(monkeypatch):
    calls = {"n": 0}

    def bad(bbox=None, tags=None):
        calls["n"] += 1
        raise _overpass_error(400)

    monkeypatch.setattr(osm.ox, "features_from_bbox", bad)
    monkeypatch.setattr(osm.time, "sleep", lambda s: None)
    with pytest.raises(Exception):
        osm.download_features((0, 0, 1, 1), {"building": True}, max_retries=3, backoff_base=1)
    assert calls["n"] == 1


def test_download_features_rotates_endpoints(monkeypatch):
    original = osm.ox.settings.overpass_url
    monkeypatch.setenv(osm.OVERPASS_URL_ENV_VAR, "http://a/api, http://b/api")
    seen = []

    def flaky(bbox=None, tags=None):
        seen.append(osm.ox.settings.overpass_url)
        if len(seen) == 1:
            raise _overpass_error(502)
        return "gdf"

    try:
        osm.configure_osm_client()
        monkeypatch.setattr(osm.ox, "features_from_bbox", flaky)
        monkeypatch.setattr(osm.time, "sleep", lambda s: None)
        osm.download_features((0, 0, 1, 1), {"building": True}, max_retries=2, backoff_base=1)
        assert seen == ["http://a/api", "http://b/api"]
    finally:
        monkeypatch.delenv(osm.OVERPASS_URL_ENV_VAR, raising=False)
        osm.configure_osm_client()
        osm.ox.settings.overpass_url = original
