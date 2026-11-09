"""HR post-processing: single-class tile filter."""

import os

import numpy as np
from PIL import Image, ImageColor

from GenCP_data_preparation.HR import config
from GenCP_data_preparation.HR.single_class_tiles import (
    build_palette,
    dominant_class_ratio,
    process_split,
)

BROWN = ImageColor.getrgb("brown")
RESIDENTIAL = ImageColor.getrgb(config.color_dict["light_gray"])
FARMLAND = ImageColor.getrgb(config.color_dict["light_green"])


def _tile(parts, size=100):
    """(H, W, 3) tile made of horizontal bands: parts = [(rgb, n_rows), ...]."""
    rows = [np.tile(np.array(rgb, np.uint8), (n, size, 1)) for rgb, n in parts]
    return np.concatenate(rows, axis=0)


def _save(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path)


# --- H5: palette and tolerance ---------------------------------------------

def test_urban_tile_is_not_suspect_nor_single_class():
    # 60 % buildings + 40 % residential: two classes, kept at threshold 0.99
    ratio, dominant, n_classes = dominant_class_ratio(
        _tile([(BROWN, 60), (RESIDENTIAL, 40)]), build_palette())
    assert ratio is not None
    assert dominant == BROWN
    assert abs(ratio - 0.6) < 1e-9
    assert n_classes == 2


def test_lossy_colours_match_within_tolerance():
    near = tuple(c - 2 for c in FARMLAND)  # e.g. JPEG-compressed farmland
    arr = _tile([(near, 70), (RESIDENTIAL, 30)])
    ratio, dominant, _ = dominant_class_ratio(arr, build_palette())
    assert dominant == FARMLAND and abs(ratio - 0.7) < 1e-9
    # exact matching still available
    assert dominant_class_ratio(arr, build_palette(), tolerance=0)[0] is None


# --- H6: deletion covers every role folder ----------------------------------

def test_single_class_tile_removed_from_all_folders(tmp_path):
    split = tmp_path / "train"
    mono = _tile([(FARMLAND, 100)])
    mixed = _tile([(FARMLAND, 50), (BROWN, 50)])
    for role in ("A", "B", "TOPO"):
        _save(str(split / role / "mono.png"), mono)
        _save(str(split / role / "mixed.png"), mixed)

    process_split(split, threshold=0.99, dry_run=False, palette=build_palette())

    for role in ("A", "B", "TOPO"):
        assert sorted(os.listdir(split / role)) == ["mixed.png"]


def test_seasonal_split_removes_both_seasons(tmp_path):
    split = tmp_path / "train"
    mono = _tile([(FARMLAND, 100)])
    mixed = _tile([(FARMLAND, 50), (BROWN, 50)])
    folders = ("A_summer", "B_summer", "A_winter", "B_winter")
    for folder in folders:
        season = folder.split("_")[1]
        _save(str(split / folder / f"T30TXT_tile_0000_0000_{season}.png"), mono)
        _save(str(split / folder / f"T30TXT_tile_0001_0000_{season}.png"), mixed)

    process_split(split, threshold=0.99, dry_run=False, palette=build_palette())

    for folder in folders:
        season = folder.split("_")[1]
        assert os.listdir(split / folder) == [f"T30TXT_tile_0001_0000_{season}.png"]


def test_dry_run_deletes_nothing(tmp_path):
    split = tmp_path / "test"
    for role in ("A", "B"):
        _save(str(split / role / "mono.png"), _tile([(FARMLAND, 100)]))

    process_split(split, threshold=0.99, dry_run=True, palette=build_palette())

    assert os.listdir(split / "A") == ["mono.png"]
    assert os.listdir(split / "B") == ["mono.png"]
