"""HR train/test split helpers: the files they pair must actually match."""

import os

import numpy as np
import rasterio
from rasterio.transform import from_origin

from GenCP_data_preparation.HR.create_seasonal_train_test_split import (
    split_seasonal,
    tile_key,
)
from GenCP_data_preparation.HR.create_train_test_split import split_topography
from GenCP_data_preparation.HR.topography_processing import extract_DEM_crop


def _write_raster(path, x0, y0, size, res, fill=100):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    profile = dict(
        driver="GTiff", height=size, width=size, count=1, dtype="uint8",
        crs="EPSG:32631", transform=from_origin(x0, y0, res, res),
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.full((1, size, size), fill, "uint8"))


def _touch_image(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x")


def test_dem_crop_keeps_chip_name(tmp_path):
    chip = str(tmp_path / "A" / "31TCH_0529_00.tif")
    dem = str(tmp_path / "dem.tif")
    _write_raster(chip, 400000, 5000000, 32, 10)
    _write_raster(dem, 399000, 5001000, 600, 10)
    os.makedirs(tmp_path / "TOPO")

    out = extract_DEM_crop(chip, dem, str(tmp_path / "TOPO"))

    assert os.path.basename(out) == "31TCH_0529_00.tif"


def test_topography_split_pairs_dem_crops(tmp_path):
    product = tmp_path / "in" / "S2_product"
    dem = str(tmp_path / "dem.tif")
    _write_raster(dem, 399000, 5001000, 600, 10)
    os.makedirs(product / "TOPO")
    for name, x0 in (("31TCH_0529_00", 400000), ("31TCH_0530_00", 400320)):
        a = str(product / "A" / f"{name}.tif")
        _write_raster(a, x0, 5000000, 32, 10)
        _write_raster(str(product / "B" / f"{name}.tif"), x0, 5000000, 32, 10)
        extract_DEM_crop(a, dem, str(product / "TOPO"))

    out = tmp_path / "out"
    split_topography(str(tmp_path / "in"), str(out), "A", "B", "TOPO",
                     test_ratio=0.5, seed=0, symlink=False)

    placed = sorted(p.name for p in out.rglob("*.tif"))
    assert len(placed) == 6  # 2 tiles x (A, B, TOPO)


def test_tile_key_strips_season_suffix():
    assert tile_key("T30TXT_tile_0000_0000_summer") == "T30TXT_tile_0000_0000"
    assert tile_key("T30TXT_tile_0000_0000_Winter") == "T30TXT_tile_0000_0000"
    assert tile_key("tile_0001_0001") == "tile_0001_0001"
    assert tile_key("summer_tile_0001") == "summer_tile_0001"


def test_seasonal_split_pairs_season_suffixed_names(tmp_path):
    for season in ("summer", "winter"):
        for role in ("A", "B"):
            for i in range(3):
                _touch_image(str(tmp_path / season / "prod" / role
                                 / f"T30TXT_tile_000{i}_0000_{season}.png"))

    out = tmp_path / "out"
    split_seasonal(str(tmp_path / "summer"), str(tmp_path / "winter"), str(out),
                   "A", "B", test_ratio=0.34, seed=0, symlink=False)

    for phase in ("train", "test"):
        keys = {
            folder: {tile_key(os.path.splitext(f)[0])
                     for f in os.listdir(out / phase / folder)}
            for folder in ("A_summer", "B_summer", "A_winter", "B_winter")
        }
        # each phase holds the same tiles in all four folders
        assert len({frozenset(k) for k in keys.values()}) == 1
    assert len(os.listdir(out / "test" / "A_summer")) == 1
    assert len(os.listdir(out / "train" / "A_summer")) == 2


def test_seasonal_split_still_pairs_plain_names(tmp_path):
    for season in ("summer", "winter"):
        for role in ("A", "B"):
            _touch_image(str(tmp_path / season / "prod" / role / "tile_0001_0001.png"))

    out = tmp_path / "out"
    split_seasonal(str(tmp_path / "summer"), str(tmp_path / "winter"), str(out),
                   "A", "B", test_ratio=0.2, seed=0, symlink=False)

    assert os.listdir(out / "test" / "B_winter") == ["tile_0001_0001.png"]
