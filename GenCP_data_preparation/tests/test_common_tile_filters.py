"""Tests for GenCP_data_preparation.common.tile_filters."""

import numpy as np
import pytest

from GenCP_data_preparation.common import tile_filters as tf


def test_black_pixel_ratio_all_black():
    assert tf.black_pixel_ratio(np.zeros((3, 4, 4), np.uint8)) == 1.0


def test_black_pixel_ratio_none_black():
    assert tf.black_pixel_ratio(np.ones((3, 4, 4), np.uint8)) == 0.0


def test_black_pixel_ratio_half():
    data = np.zeros((3, 2, 2), np.uint8)
    data[:, 0, :] = 255  # top row non-black -> 2 of 4 pixels black
    assert tf.black_pixel_ratio(data) == pytest.approx(0.5)


def test_black_pixel_ratio_pixel_black_only_if_all_bands_zero():
    data = np.zeros((3, 1, 1), np.uint8)
    data[0, 0, 0] = 1  # one band non-zero -> pixel is NOT black
    assert tf.black_pixel_ratio(data) == 0.0


def test_black_pixel_ratio_2d():
    data = np.array([[0, 0], [1, 0]], np.uint8)  # 3 of 4 black
    assert tf.black_pixel_ratio(data) == pytest.approx(0.75)


def test_is_all_black():
    assert tf.is_all_black(np.zeros((3, 5, 5), np.uint8)) is True
    nz = np.zeros((3, 5, 5), np.uint8)
    nz[1, 2, 2] = 7
    assert tf.is_all_black(nz) is False


def test_is_all_black_matches_ratio_extreme():
    z = np.zeros((3, 4, 4), np.uint8)
    assert tf.is_all_black(z) == (tf.black_pixel_ratio(z) == 1.0)
