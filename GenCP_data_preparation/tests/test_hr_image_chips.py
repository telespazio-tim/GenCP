"""HR chip extraction: GRI cell selection and full-tile chip names."""

import os

import numpy as np

from GenCP_data_preparation.HR.image_chips_extractor import gri_cells


# --- H12: GRI 1-degree cells -------------------------------------------------

def test_gri_cells_negative_longitude_uses_floor():
    # zone 30 tile west of Greenwich: lon -1.6 .. -0.3 lies in cells W002 and W001
    assert gri_cells(-1.6, 43.2, -0.3, 43.9) == [(43, -2), (43, -1)]


def test_gri_cells_include_middle_cells():
    # a 1.4 deg wide box from 0.9 to 2.3 covers cells 0, 1 and 2
    assert gri_cells(0.9, 44.1, 2.3, 44.9) == [(44, 0), (44, 1), (44, 2)]


def test_gri_cells_across_latitude_boundary():
    assert gri_cells(3.2, 43.8, 3.9, 44.3) == [(43, 3), (44, 3)]


# --- H7: full-tile chip names carry the MGRS tile -----------------------------

def _fake_s2_product(root, name, x0):
    import rasterio
    from rasterio.transform import from_origin

    band_dir = root / name / "GRANULE" / "L2A" / "IMG_DATA" / "R10m"
    os.makedirs(band_dir)
    # GeoTIFF content under the .jp2 name the resolver looks for (GDAL reads by content)
    with rasterio.open(band_dir / "X_TCI_10m.jp2", "w", driver="GTiff", width=512, height=512,
                       count=1, dtype="uint8", crs="EPSG:32631",
                       transform=from_origin(x0, 4900000, 10, 10)) as dst:
        dst.write(np.full((1, 512, 512), 120, np.uint8))
    return str(root / name)


def test_full_tile_chips_of_two_tiles_do_not_collide(tmp_path):
    from GenCP_data_preparation.HR.image_chips_extractor import tile_s2_full_image

    out = tmp_path / "out"
    names = {}
    for mgrs, x0 in (("31TCJ", 300000), ("31TDJ", 399960)):
        product = _fake_s2_product(
            tmp_path / "in", f"S2A_MSIL2A_20240601T104021_N0510_R008_T{mgrs}_20240601T140000", x0)
        assert tile_s2_full_image(product, "TCI", 2560.0, str(out)) == 4
        names[mgrs] = {f for p in out.glob(f"*T{mgrs}*/TCI/S2_crop_extended") for f in os.listdir(p)}

    assert names["31TCJ"] == {f"T31TCJ_tile_000{c}_000{r}.tif" for c in (0, 1) for r in (0, 1)}
    assert not names["31TCJ"] & names["31TDJ"]
