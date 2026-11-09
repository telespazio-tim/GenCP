"""Test setup shared across the data-prep test suite."""

import os


def _ensure_proj_data():
    """Point GDAL/PROJ at its data dir when the env isn't `conda activate`d.

    Bare-interpreter invocation (e.g. `<env>/bin/python -m pytest`) doesn't run
    conda's activation scripts, so PROJ_DATA/PROJ_LIB may be unset and osr
    reprojection fails to open proj.db. Derive it from pyproj if needed.
    """
    if os.environ.get("PROJ_DATA") or os.environ.get("PROJ_LIB"):
        return
    try:
        import pyproj

        data_dir = pyproj.datadir.get_data_dir()
    except Exception:
        return
    if data_dir:
        os.environ["PROJ_DATA"] = data_dir
        os.environ["PROJ_LIB"] = data_dir


_ensure_proj_data()
