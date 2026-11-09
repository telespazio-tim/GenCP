"""Tests for GenCP_data_preparation.common.raster_io."""

import numpy as np
import pytest
from osgeo import gdal

from GenCP_data_preparation.common import raster_io


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (np.uint8, gdal.GDT_Byte),
        (np.uint16, gdal.GDT_UInt16),
        (np.int16, gdal.GDT_Int16),
        (np.uint32, gdal.GDT_UInt32),
        (np.int32, gdal.GDT_Int32),
        (np.float32, gdal.GDT_Float32),
        (np.float64, gdal.GDT_Float64),
    ],
)
def test_gdal_dtype(dtype, expected):
    assert raster_io.gdal_dtype(np.zeros((1, 1), dtype)) == expected


def test_gdal_dtype_fallback():
    assert raster_io.gdal_dtype(np.zeros((1, 1), np.int8)) == gdal.GDT_Float32


def test_write_geotiff_roundtrip(tmp_path):
    path = str(tmp_path / "out.tif")
    data = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)  # (bands, H, W)
    gt = (400000.0, 10.0, 0.0, 5000000.0, 0.0, -10.0)
    srs_wkt = _utm31_wkt()

    raster_io.write_geotiff(path, data, gt, srs_wkt, n_bands=3)

    ds = gdal.Open(path)
    assert ds.RasterXSize == 5 and ds.RasterYSize == 4   # non-square dims derived from data
    assert ds.RasterCount == 3
    assert ds.GetGeoTransform() == pytest.approx(gt)
    assert ds.GetRasterBand(1).DataType == gdal.GDT_Byte
    for b in range(3):
        np.testing.assert_array_equal(ds.GetRasterBand(b + 1).ReadAsArray(), data[b])
    ds = None


def test_write_geotiff_rejects_2d(tmp_path):
    with pytest.raises(ValueError):
        raster_io.write_geotiff(
            str(tmp_path / "bad.tif"), np.zeros((4, 5), np.uint8),
            (0, 1, 0, 0, 0, -1), _utm31_wkt(), n_bands=1,
        )


def _utm31_wkt():
    from osgeo import osr
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32631)
    return srs.ExportToWkt()
