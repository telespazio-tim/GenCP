"""VHR (very-high-resolution) OSM-styled dataset preparation.

Standalone scripts (run directly / via the notebook and run.bat) that build paired
raster + styled-OSM-mask tiles. Shared plumbing lives in
:mod:`GenCP_data_preparation.common`; the scripts add the repo root to ``sys.path``
so those imports resolve when executed directly.
"""
