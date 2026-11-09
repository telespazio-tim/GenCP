"""Shared GeoTIFF writing helpers used by both pipelines."""

import numpy as np
from osgeo import gdal

# numpy dtype -> GDAL data type
_NP_TO_GDAL = {
    np.uint8: gdal.GDT_Byte,
    np.uint16: gdal.GDT_UInt16,
    np.int16: gdal.GDT_Int16,
    np.uint32: gdal.GDT_UInt32,
    np.int32: gdal.GDT_Int32,
    np.float32: gdal.GDT_Float32,
    np.float64: gdal.GDT_Float64,
}


def gdal_dtype(arr: np.ndarray) -> int:
    """Map a numpy array's dtype to the matching GDAL data type.

    Falls back to ``gdal.GDT_Float32`` for anything unmapped.
    """
    return _NP_TO_GDAL.get(arr.dtype.type, gdal.GDT_Float32)


def write_geotiff(
    path,
    data: np.ndarray,
    geotransform,
    projection,
    n_bands: int = None,
    dtype: int = None,
):
    """Write a band-first numpy array to a GeoTIFF.

    Args:
        path: Output path.
        data: Array shaped ``(bands, height, width)``.
        geotransform: 6-tuple GDAL geotransform.
        projection: WKT projection string.
        n_bands: Number of bands to write. Defaults to ``data.shape[0]``.
        dtype: GDAL data type. Defaults to :func:`gdal_dtype` of ``data``.

    The raster width/height are derived from ``data`` so non-square tiles work.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected (bands, height, width) array, got shape {data.shape}")

    bands = n_bands if n_bands is not None else data.shape[0]
    height, width = data.shape[1], data.shape[2]
    dt = dtype if dtype is not None else gdal_dtype(data)

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, width, height, bands, dt)
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)
    for b in range(bands):
        ds.GetRasterBand(b + 1).WriteArray(data[b])
    ds.FlushCache()
    ds = None
