"""Tests for the shared BaseConfig and HR's use of it."""

from pathlib import Path

import pytest

from GenCP_data_preparation.common.config import BaseConfig
from GenCP_data_preparation.HR.config import HRPipelineConfig


def test_base_config_coerces_paths():
    cfg = BaseConfig(input_dir="in/dir", output_dir="out/dir")
    assert isinstance(cfg.input_dir, Path) and isinstance(cfg.output_dir, Path)
    assert cfg.input_dir == Path("in/dir")


def test_base_config_defaults():
    cfg = BaseConfig()
    assert cfg.input_dir == Path(".") and cfg.output_dir == Path(".")


def test_hr_config_is_base_subclass():
    assert issubclass(HRPipelineConfig, BaseConfig)


def test_hr_config_inherits_path_coercion():
    cfg = HRPipelineConfig(input_dir="a/b", output_dir="c/d", band="TCI")
    assert isinstance(cfg.input_dir, Path) and isinstance(cfg.output_dir, Path)
    # subclass __post_init__ still runs its own coercion + properties
    assert cfg.s2_crop_folder == "S2_crop_extended"


def test_hr_config_band_validation_still_enforced():
    with pytest.raises(ValueError):
        HRPipelineConfig(band="B99")


def test_hr_config_aoi_mode_skips_band_check():
    cfg = HRPipelineConfig(aoi_path="aoi.geojson", band="B99")  # band ignored in AOI mode
    assert cfg.is_aoi_mode is True
