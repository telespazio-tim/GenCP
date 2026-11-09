"""Shared pixel-grid tiling primitives.

The HR pipeline tiles in *metric* units (see
:func:`GenCP_data_preparation.HR.geo_utils.generate_chip_grid_from_bounds`),
while the VHR pipeline tiles a raster in *pixel* units. The two abstractions are
kept separate on purpose; this module holds the pixel-based helpers shared by the
VHR path (and available to any future HR pixel-tiling path).
"""

from typing import Iterator, Tuple


def tile_geotransform(src_gt, x_off: int, y_off: int) -> Tuple[float, float, float, float, float, float]:
    """Derive a tile's GDAL geotransform from its parent's geotransform.

    Handles the full affine (including rotation terms), so a tile whose top-left
    pixel is ``(x_off, y_off)`` in the source gets the correct origin.

    Args:
        src_gt: The 6-tuple GDAL geotransform of the source raster
            ``(ox, pw, rx, oy, ry, ph)``.
        x_off: Column offset (pixels) of the tile's top-left corner.
        y_off: Row offset (pixels) of the tile's top-left corner.

    Returns:
        The tile's 6-tuple geotransform (pixel sizes/rotations unchanged).
    """
    ox, pw, rx, oy, ry, ph = src_gt
    tox = ox + x_off * pw + y_off * rx
    toy = oy + x_off * ry + y_off * ph
    return (tox, pw, rx, toy, ry, ph)


def pixel_tile_grid(
    width: int,
    height: int,
    tile_size: int,
    overlap: int = 0,
    drop_partial: bool = True,
) -> Iterator[Tuple[int, int, int, int, int, int]]:
    """Yield tile windows over a raster of ``width`` x ``height`` pixels.

    Iterates row-major (top-to-bottom, left-to-right), matching the VHR tiler.

    Args:
        width: Raster width in pixels.
        height: Raster height in pixels.
        tile_size: Square tile side length in pixels.
        overlap: Overlap between adjacent tiles in pixels (default 0).
        drop_partial: If True, tiles that would extend past the raster edge are
            skipped (floor-division behaviour, as VHR uses). If False, partial
            edge tiles are yielded, clipped to the raster.

    Yields:
        ``(col, row, x_off, y_off, win_w, win_h)`` per tile.
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    step = tile_size - overlap
    if step <= 0:
        raise ValueError(f"overlap ({overlap}) must be smaller than tile_size ({tile_size})")

    row = 0
    y = 0
    while y < height:
        if drop_partial and y + tile_size > height:
            break
        col = 0
        x = 0
        while x < width:
            if drop_partial and x + tile_size > width:
                break
            win_w = min(tile_size, width - x)
            win_h = min(tile_size, height - y)
            yield (col, row, x, y, win_w, win_h)
            col += 1
            x += step
        row += 1
        y += step
