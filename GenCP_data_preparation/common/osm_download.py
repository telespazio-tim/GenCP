"""Shared OSM acquisition.

This module is the single place where the osmnx features API is called, so the
osmnx version and argument convention live in one spot. It targets **osmnx >= 2.0**,
whose ``features_from_bbox(bbox, tags)`` expects ``bbox`` in the canonical
``(west, south, east, north)`` order (``(left, bottom, right, top)``) used
throughout :mod:`GenCP_data_preparation.common`.
"""

import logging
import os
import random
import re
import time

import osmnx as ox
import requests
from osmnx._errors import ResponseStatusCodeError

logger = logging.getLogger(__name__)

# Overpass endpoints reject osmnx's default User-Agent with HTTP 406, so both
# pipelines present a descriptive one. Overridable via the environment (useful
# behind stricter endpoints that want a browser-like UA) without editing code.
DEFAULT_USER_AGENT = "GenCP-data-prep/1.0"
USER_AGENT_ENV_VAR = "GENCP_OSM_USER_AGENT"
OVERPASS_URL_ENV_VAR = "GENCP_OVERPASS_URL"

# Retry policy for transient Overpass failures (server at capacity, gateway
# errors, dropped connections). osmnx itself only retries 429/504, so a 502
# "all query workers are busy" would otherwise abort the run immediately.
MAX_RETRIES_ENV_VAR = "GENCP_OSM_MAX_RETRIES"
BACKOFF_BASE_ENV_VAR = "GENCP_OSM_BACKOFF_BASE"
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_BASE = 30.0  # seconds; doubles each retry: 30, 60, 120, 240, 480
MAX_BACKOFF = 600.0
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}

_STATUS_RE = re.compile(r"responded:\s*(\d{3})")

# Endpoints rotated through on transient failures. Filled by
# configure_osm_client() when GENCP_OVERPASS_URL holds a comma-separated list.
_overpass_urls = []


def configure_osm_client(user_agent: str = None, overpass_url: str = None) -> str:
    """Set the osmnx HTTP User-Agent / referer (and optionally the Overpass URL).

    User-Agent resolution order: explicit ``user_agent`` argument, then the
    ``GENCP_OSM_USER_AGENT`` environment variable, then :data:`DEFAULT_USER_AGENT`.

    Overpass endpoint: explicit ``overpass_url`` argument, else the
    ``GENCP_OVERPASS_URL`` environment variable if set (e.g. a mirror or a local
    Overpass instance). Either may be a comma-separated list of endpoints: the
    first is used, and :func:`download_features` rotates to the next one after
    a transient failure. When neither is given, osmnx's default endpoint is left
    unchanged.

    Returns the applied User-Agent string.
    """
    ua = user_agent or os.environ.get(USER_AGENT_ENV_VAR) or DEFAULT_USER_AGENT
    ox.settings.http_user_agent = ua
    ox.settings.http_referer = ua

    url = overpass_url or os.environ.get(OVERPASS_URL_ENV_VAR)
    _overpass_urls.clear()
    if url:
        _overpass_urls.extend(u.strip() for u in url.split(",") if u.strip())
        ox.settings.overpass_url = _overpass_urls[0]

    return ua


# Apply on import so every download_features() call uses an accepted User-Agent.
configure_osm_client()


def is_transient_error(error) -> bool:
    """True if ``error`` is an Overpass/network failure worth retrying later.

    Covers dropped connections and timeouts, and HTTP 429/5xx responses that
    osmnx surfaces as :class:`ResponseStatusCodeError` (e.g. 502 "server at
    capacity"). Other errors (bad query, no data, ...) are not transient.
    """
    if isinstance(error, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    if isinstance(error, ResponseStatusCodeError):
        match = _STATUS_RE.search(str(error))
        # No parsable status: osmnx only raises this for non-2xx responses, so
        # assume the server side is at fault and let the retry decide.
        return match is None or int(match.group(1)) in TRANSIENT_STATUS_CODES
    return False


def _rotate_overpass_url() -> None:
    """Switch osmnx to the next configured Overpass endpoint, if several are set."""
    if len(_overpass_urls) < 2:
        return
    try:
        idx = _overpass_urls.index(ox.settings.overpass_url)
    except ValueError:
        idx = -1
    ox.settings.overpass_url = _overpass_urls[(idx + 1) % len(_overpass_urls)]
    logger.warning("Switching Overpass endpoint to %s", ox.settings.overpass_url)


def download_features(bbox, tags, max_retries: int = None, backoff_base: float = None):
    """Download OSM features within a canonical ``(west, south, east, north)`` bbox.

    Transient failures (see :func:`is_transient_error`) are retried with
    exponential backoff plus jitter, rotating Overpass endpoints when several
    are configured. Responses already fetched are served from the osmnx cache,
    so a retry only re-requests the sub-queries that failed.

    Args:
        bbox: Bounding box ``(west, south, east, north)`` in EPSG:4326.
        tags: osmnx tag filter dict (e.g. ``{"building": True, "highway": True}``).
        max_retries: Retries after the first attempt. Defaults to
            ``GENCP_OSM_MAX_RETRIES`` or :data:`DEFAULT_MAX_RETRIES`.
        backoff_base: First wait in seconds, doubled each retry (capped at
            :data:`MAX_BACKOFF`). Defaults to ``GENCP_OSM_BACKOFF_BASE`` or
            :data:`DEFAULT_BACKOFF_BASE`.

    Returns:
        A GeoDataFrame of features in EPSG:4326.

    Raises:
        osmnx._errors.InsufficientResponseError: If Overpass returns nothing.
        The last transient error, once retries are exhausted.
    """
    if max_retries is None:
        max_retries = int(os.environ.get(MAX_RETRIES_ENV_VAR, DEFAULT_MAX_RETRIES))
    if backoff_base is None:
        backoff_base = float(os.environ.get(BACKOFF_BASE_ENV_VAR, DEFAULT_BACKOFF_BASE))

    for attempt in range(max_retries + 1):
        try:
            return ox.features_from_bbox(bbox=bbox, tags=tags)
        except Exception as e:
            if not is_transient_error(e) or attempt == max_retries:
                raise
            wait = min(backoff_base * 2 ** attempt, MAX_BACKOFF)
            wait *= random.uniform(0.8, 1.2)
            logger.warning(
                "Overpass request failed (%s: %s). Retry %d/%d in %.0f s",
                e.__class__.__name__, str(e).splitlines()[0][:200],
                attempt + 1, max_retries, wait,
            )
            time.sleep(wait)
            _rotate_overpass_url()


def split_bbox(bbox, n: int = 2):
    """Split a canonical ``(west, south, east, north)`` bbox into ``n*n`` tiles.

    Used to bound Overpass response size by querying sub-bboxes. Tiles are
    produced column-major (matching the VHR quadrant-split it replaces) and
    their union is exactly the input bbox.

    Args:
        bbox: ``(west, south, east, north)``.
        n: Number of divisions per axis (``n*n`` sub-bboxes).

    Returns:
        List of ``(west, south, east, north)`` sub-bboxes.
    """
    west, south, east, north = bbox
    dx = (east - west) / n
    dy = (north - south) / n
    parts = []
    for ix in range(n):
        for iy in range(n):
            parts.append(
                (west + ix * dx, south + iy * dy, west + (ix + 1) * dx, south + (iy + 1) * dy)
            )
    return parts


def combine_tags(tag_dicts):
    """Merge several osmnx tag dicts into one combined filter.

    ``True`` (match-any-value) dominates specific values for the same key;
    otherwise the specific values are unioned and sorted.

    Args:
        tag_dicts: Iterable of tag dicts.

    Returns:
        A single merged tag dict.
    """
    combined = {}
    for spec in tag_dicts:
        for key, value in spec.items():
            if key not in combined:
                combined[key] = value
                continue
            if combined[key] is True or value is True:
                combined[key] = True
                continue
            old_values = combined[key] if isinstance(combined[key], list) else [combined[key]]
            new_values = value if isinstance(value, list) else [value]
            combined[key] = sorted(set(old_values + new_values))
    return combined
