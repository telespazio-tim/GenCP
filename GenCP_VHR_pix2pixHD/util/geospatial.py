"""Georeference pix2pixHD outputs and mosaic tiles by source image.

Adapted from the supplied georeferencing.ipynb and grouped merge.ipynb code.
GDAL is imported only when raster processing is requested.
"""
from pathlib import Path
from collections import defaultdict
import re
import os
import ntpath


def _gdal():
    try:
        from osgeo import gdal
    except ImportError as exc:
        raise ImportError('GDAL Python bindings are required for georeferencing and merging. '
                          'Use a notebook kernel with osgeo.gdal installed.') from exc
    gdal.UseExceptions()
    return gdal


def _files(folder, extensions):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in extensions)


def build_reference_index(dataset_dir):
    """Index GeoTIFFs named <prefix>_mask_<id> or <prefix>_patch_<id>."""
    index = {}
    for path in _files(dataset_dir, {'.tif', '.tiff'}):
        match = re.fullmatch(r'(.+)_(?:mask|patch)_(\d+)', path.stem, re.IGNORECASE)
        if match is None:
            continue
        key = match.group(1), match.group(2)
        if key in index:
            raise ValueError(f'Ambiguous reference key {key}: {index[key]} and {path}')
        index[key] = path
    if not index:
        raise ValueError('No reference GeoTIFFs match <prefix>_mask_<id> or <prefix>_patch_<id>: ' + str(dataset_dir))
    return index


def normalize_predict_prefix(prefix):
    """Preserve the prefix normalization rules from the supplied notebook."""
    for suffix in ['_mask_modified', '_modified_mask', '_input_label', '_synthesized_image']:
        if prefix.lower().endswith(suffix):
            prefix = prefix[:-len(suffix)]
    return prefix


def parse_predict_name(path):
    """Return reference candidate keys and output kind, or None if unmatched."""
    match = re.fullmatch(r'(.+)_mask_(\d+)_(input_label|synthesized_image)',
                         Path(path).stem, re.IGNORECASE)
    if match is None:
        return None
    raw, patch_id, kind = match.groups()
    keys = [(raw, patch_id)]
    normalized = normalize_predict_prefix(raw)
    if normalized != raw:
        keys.append((normalized, patch_id))
    return keys, kind.lower()


def georeference_jpg_with_ref(jpg_path, ref_tif_path, out_tif_path,
                              resize_to_ref=True, overwrite=False):
    """Copy reference CRS and affine transform; optionally resize to its grid.

    This assumes that the JPG and reference cover the same ground footprint.
    Assigning metadata does not correct misalignment or model-generated distortions.
    """
    gdal = _gdal()
    out_tif_path = Path(out_tif_path)
    if out_tif_path.exists() and not overwrite:
        raise FileExistsError(f'Output already exists: {out_tif_path}. Enable overwrite intentionally.')
    ref = src = out = None
    try:
        ref = gdal.Open(str(ref_tif_path), gdal.GA_ReadOnly)
        src = gdal.Open(str(jpg_path), gdal.GA_ReadOnly)
        if ref is None or src is None:
            raise RuntimeError(f'Cannot open source or reference: {jpg_path}, {ref_tif_path}')
        projection = ref.GetProjection()
        transform = ref.GetGeoTransform(can_return_null=True)
        if not projection or transform is None:
            raise ValueError(f'Reference lacks CRS or affine geotransform: {ref_tif_path}')
        mismatch = (src.RasterXSize, src.RasterYSize) != (ref.RasterXSize, ref.RasterYSize)
        if mismatch and not resize_to_ref:
            raise ValueError('Image sizes differ; copying the reference transform unchanged would '
                             'give a different footprint. Enable resize_to_ref or inspect the data.')
        out_tif_path.parent.mkdir(parents=True, exist_ok=True)
        options = ['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
        if mismatch:
            print(f'[RESIZE] {Path(jpg_path).name}: {src.RasterXSize}x{src.RasterYSize} '
                  f'-> {ref.RasterXSize}x{ref.RasterYSize}')
            algorithm = 'near' if Path(jpg_path).stem.lower().endswith('_input_label') else 'bilinear'
            out = gdal.Translate(str(out_tif_path), src, format='GTiff',
                                 width=ref.RasterXSize, height=ref.RasterYSize,
                                 resampleAlg=algorithm, creationOptions=options)
        else:
            out = gdal.GetDriverByName('GTiff').CreateCopy(str(out_tif_path), src, strict=0, options=options)
        if out is None:
            raise RuntimeError(f'Cannot create output: {out_tif_path}')
        out.SetProjection(projection)
        out.SetGeoTransform(transform)
        out.FlushCache()
    finally:
        out = src = ref = None
    return out_tif_path


def georeference_results(dataset_dir, predict_dir, output_dir, only='all',
                         resize_to_ref=True, overwrite=False):
    """Match selected test outputs to references, then create georeferenced tiles."""
    if only not in ('all', 'input_label', 'synthesized_image'):
        raise ValueError('only must be all, input_label or synthesized_image')
    refs = build_reference_index(dataset_dir)
    plan, missing, skipped = [], [], []
    for jpg in _files(predict_dir, {'.jpg', '.jpeg'}):
        parsed = parse_predict_name(jpg)
        if parsed is None:
            skipped.append(jpg.name)
            continue
        keys, kind = parsed
        if only != 'all' and kind != only:
            continue
        reference = next((refs[k] for k in keys if k in refs), None)
        if reference is None:
            missing.append(f'{jpg.name}: tried {keys}')
            continue
        target = Path(output_dir) / kind / (jpg.stem + '.tif')
        plan.append((jpg, reference, target))
    if missing:
        raise ValueError('Missing references; no outputs written:\n' + '\n'.join(missing))
    if not plan:
        raise ValueError('No selected JPG outputs match <prefix>_mask_<id>_<kind>.jpg')
    targets = [item[2] for item in plan]
    if len(set(targets)) != len(targets):
        raise ValueError('Multiple input files map to the same output filename')
    existing = [p for p in targets if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f'{len(existing)} outputs already exist; first: {existing[0]}')
    print(f'References: {len(refs)}; selected outputs: {len(plan)}; unmatched names: {len(skipped)}')
    for name in skipped:
        print('[SKIP NAME]', name)
    outputs = []
    for jpg, reference, target in plan:
        outputs.append(georeference_jpg_with_ref(jpg, reference, target, resize_to_ref, overwrite))
        print(f'[OK] {jpg.name} <- {reference.name}')
    print('Georeferencing complete:', len(outputs), 'tiles in', Path(output_dir))
    return outputs


def parse_georef_tile_name(path):
    """Return raw image prefix, numeric patch ID and output kind."""
    match = re.fullmatch(r'(.+)_mask_(\d+)_(input_label|synthesized_image)',
                         Path(path).stem, re.IGNORECASE)
    if match is None:
        return None
    prefix, patch_id, kind = match.groups()
    return prefix, int(patch_id), kind.lower()



def _windows_extended_path(path):
    """Normalize a Windows path and add the extended-length prefix once."""
    value = os.fspath(path).replace('/', '\\')
    if value.startswith('\\\\?\\'):
        return value
    value = ntpath.abspath(value)
    if value.startswith('\\\\'):
        return '\\\\?\\UNC\\' + value[2:]
    return '\\\\?\\' + value


def _merge_path(path):
    """Use extended paths for Windows filesystem checks and GDAL mosaic I/O."""
    return Path(_windows_extended_path(path)) if os.name == 'nt' else Path(path)


def merge_tiles_with_gdal_warp(tile_paths, output_file, resample_alg='near', overwrite=False):
    """Mosaic tiles using the supplied notebook's GDAL Warp settings."""
    gdal = _gdal()
    output_file = _merge_path(output_file)
    tile_paths = [_merge_path(p) for p in tile_paths]
    if not tile_paths:
        raise ValueError('Cannot merge an empty tile list')
    if output_file.exists() and not overwrite:
        raise FileExistsError(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    options = gdal.WarpOptions(format='GTiff', resampleAlg=resample_alg, multithread=True,
                              creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER'])
    ds = gdal.Warp(str(output_file), [str(p) for p in tile_paths], options=options)
    if ds is None:
        raise RuntimeError(f'Cannot create mosaic: {output_file}')
    ds.FlushCache()
    ds = None
    return output_file


def merge_georeferenced_tiles(tile_paths, merged_output_dir, input_resample='near',
                               synthesized_resample='bilinear', overwrite=False):
    """Merge only the supplied tiles, grouped by source image and output kind.

    Numeric patch sorting preserves the supplied notebook's overlap ordering.
    No new target CRS, pixel size, nodata value or alignment rule is imposed.
    """
    groups = defaultdict(list)
    seen = set()
    for item in tile_paths:
        path = _merge_path(item)
        if not path.is_file():
            raise FileNotFoundError(path)
        parsed = parse_georef_tile_name(path)
        if parsed is None:
            raise ValueError(f'Unexpected tile name: {path.name}')
        prefix, patch_id, kind = parsed
        key = prefix, patch_id, kind
        if key in seen:
            raise ValueError(f'Duplicate tile: {key}')
        seen.add(key)
        groups[(prefix, kind)].append((patch_id, path))
    if not groups:
        raise ValueError('No georeferenced tiles to merge')
    plan = []
    for (prefix, kind), items in sorted(groups.items()):
        output = _merge_path(Path(merged_output_dir) / f'{kind}_merged' / f'{prefix}_{kind}_merged.tif')
        if output.exists() and not overwrite:
            raise FileExistsError(output)
        plan.append((kind, [p for _, p in sorted(items)], output))
    outputs = []
    for kind, paths, output in plan:
        algorithm = input_resample if kind == 'input_label' else synthesized_resample
        outputs.append(merge_tiles_with_gdal_warp(paths, output, algorithm, overwrite))
        print(f'[MERGED] {len(paths)} tiles -> {output}')
    return outputs
