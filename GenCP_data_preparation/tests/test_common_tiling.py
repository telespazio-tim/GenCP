"""Tests for GenCP_data_preparation.common.tiling."""

import pytest

from GenCP_data_preparation.common import tiling


# ---------------------------------------------------------------------------
# tile_geotransform
# ---------------------------------------------------------------------------
def test_tile_geotransform_axis_aligned():
    src_gt = (400000.0, 10.0, 0.0, 5000000.0, 0.0, -10.0)
    # tile at pixel offset (256, 128)
    tgt = tiling.tile_geotransform(src_gt, 256, 128)
    assert tgt == (400000.0 + 256 * 10.0, 10.0, 0.0, 5000000.0 + 128 * -10.0, 0.0, -10.0)


def test_tile_geotransform_with_rotation():
    # non-zero rotation terms must be honoured
    src_gt = (100.0, 2.0, 0.5, 200.0, 0.3, -2.0)
    x_off, y_off = 4, 7
    tgt = tiling.tile_geotransform(src_gt, x_off, y_off)
    exp_ox = 100.0 + x_off * 2.0 + y_off * 0.5
    exp_oy = 200.0 + x_off * 0.3 + y_off * -2.0
    assert tgt == (exp_ox, 2.0, 0.5, exp_oy, 0.3, -2.0)


def test_tile_geotransform_matches_legacy_vhr_formula():
    """Regression: reproduce the exact inline math osm_style_processor used before."""
    gt = (612345.0, 0.5, 0.0, 4123456.0, 0.0, -0.5)
    TILE = 256
    for i in range(3):        # rows
        for j in range(3):    # cols
            px, py = j * TILE, i * TILE
            ox, pw, rx = gt[0], gt[1], gt[2]
            oy, ry, ph = gt[3], gt[4], gt[5]
            legacy = (ox + px * pw + py * rx, pw, rx, oy + px * ry + py * ph, ry, ph)
            assert tiling.tile_geotransform(gt, px, py) == legacy


# ---------------------------------------------------------------------------
# pixel_tile_grid
# ---------------------------------------------------------------------------
def test_pixel_tile_grid_exact_multiple():
    tiles = list(tiling.pixel_tile_grid(512, 512, 256))
    assert len(tiles) == 4
    # row-major order, full-size windows
    assert tiles == [
        (0, 0, 0, 0, 256, 256),
        (1, 0, 256, 0, 256, 256),
        (0, 1, 0, 256, 256, 256),
        (1, 1, 256, 256, 256, 256),
    ]


def test_pixel_tile_grid_idx_matches_vhr():
    """idx = row * n_cols + col must be contiguous 0..N-1 in row-major order."""
    width, height, tile = 768, 512, 256
    n_cols = width // tile
    tiles = list(tiling.pixel_tile_grid(width, height, tile))
    for expected_idx, (col, row, *_rest) in enumerate(tiles):
        assert row * n_cols + col == expected_idx


def test_pixel_tile_grid_drop_partial():
    # 500x300 with tile 256: floor(500/256)=1 col, floor(300/256)=1 row -> 1 tile
    tiles = list(tiling.pixel_tile_grid(500, 300, 256, drop_partial=True))
    assert tiles == [(0, 0, 0, 0, 256, 256)]


def test_pixel_tile_grid_keep_partial_clips():
    tiles = list(tiling.pixel_tile_grid(500, 300, 256, drop_partial=False))
    # 2 cols x 2 rows, edge windows clipped
    assert len(tiles) == 4
    assert (1, 0, 256, 0, 244, 256) in tiles       # right edge: 500-256=244
    assert (0, 1, 0, 256, 256, 44) in tiles        # bottom edge: 300-256=44
    assert (1, 1, 256, 256, 244, 44) in tiles


def test_pixel_tile_grid_overlap():
    # width 512, tile 256, overlap 128 -> step 128 -> offsets 0,128,256 (384 dropped: 384+256>512)
    xs = sorted({x for (_c, _r, x, _y, _w, _h) in tiling.pixel_tile_grid(512, 256, 256, overlap=128)})
    assert xs == [0, 128, 256]


def test_pixel_tile_grid_larger_than_raster():
    assert list(tiling.pixel_tile_grid(100, 100, 256)) == []


@pytest.mark.parametrize("bad", [0, -5])
def test_pixel_tile_grid_bad_tile_size(bad):
    with pytest.raises(ValueError):
        list(tiling.pixel_tile_grid(512, 512, bad))


def test_pixel_tile_grid_overlap_too_large():
    with pytest.raises(ValueError):
        list(tiling.pixel_tile_grid(512, 512, 256, overlap=256))
