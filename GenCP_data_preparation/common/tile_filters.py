"""Shared tile quality filters used by both pipelines.

These operate on raw pixel arrays (band-first ``(bands, H, W)`` or 2-D ``(H, W)``)
so they are resolution- and pipeline-agnostic. Palette-aware, semantic filtering
(e.g. HR's single-class detection) stays in the HR pipeline, which owns its OSM
colour palette.
"""

import numpy as np


def black_pixel_ratio(data: np.ndarray) -> float:
    """Fraction of fully-black pixels in a raster array.

    A pixel is "black" when every band at that position is 0. For a 2-D array a
    pixel is black when its single value is 0.

    Args:
        data: ``(bands, H, W)`` or ``(H, W)`` array.

    Returns:
        Ratio in ``[0.0, 1.0]``.
    """
    if data.ndim == 3:
        black = np.all(data == 0, axis=0)
    else:
        black = data == 0
    return float(np.count_nonzero(black) / black.size)


def is_all_black(data: np.ndarray) -> bool:
    """True when every element of the array is 0 (a fully empty/no-data tile).

    Equivalent to ``black_pixel_ratio(data) == 1.0`` but short-circuits, matching
    the ``not np.any(...)`` check the HR chip extractor has always used.
    """
    return not np.any(data)
