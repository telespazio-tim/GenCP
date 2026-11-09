"""Shared, resolution-agnostic data-preparation utilities.

This package holds logic common to the HR and VHR data-preparation pipelines.
Each pipeline keeps its own OSM class -> colour handling (HR: Python palette +
GDAL rasterize; VHR: QGIS QML symbology); only the plumbing lives here.

Modules:
    - :mod:`geo`           bounding-box extraction, UTM estimation, bounds reprojection
    - :mod:`raster_io`     GeoTIFF writing and numpy -> GDAL dtype mapping
    - :mod:`tiling`        pixel-grid tiling helpers (tile geotransform, tile windows)
    - :mod:`tile_filters`  raw-array tile quality filters (black-pixel ratio, empty)
    - :mod:`osm_download`  single osmnx (>=2.0) call point + bbox split / tag merge
"""
