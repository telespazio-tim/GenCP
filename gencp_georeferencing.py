import argparse
import os
import re

import rasterio
from rasterio.transform import Affine

# Generated images are named <reference>_<visual>.png by test.py: "_fake" with
# --model test, "_fake_B" with --model pix2pix.
GENERATED_SUFFIX_RE = re.compile(r"^(?P<ref>.+)_fake(?:_B)?\.[A-Za-z]+$")


def geoloc_generated_images(reference_path, generated_path, output_dir):
    """
    Georeference generated images based on real images metadata

    Args:
        reference_path (str): path to input reference (OSM geo-referenced raster)
        generated_path (str): path to generated image patch (not georeferenced)
        output_dir (str): path to output directory for georeferenced generated images: GenCP DB
    """

    filename = os.path.basename(reference_path)[:-4]

    with rasterio.open(generated_path) as src_test:
        test_img = src_test.read()

    out_image = f"{output_dir}/{filename}.tif"
    with rasterio.open(reference_path) as src:
        # test.py resizes inputs (e.g. 257x257 -> 256x256): rescale the pixel size so
        # the generated image covers exactly the reference footprint.
        transform = src.transform * Affine.scale(
            src.width / test_img.shape[2], src.height / test_img.shape[1]
        )
        with rasterio.open(
            out_image,
            "w",
            driver="GTiff",
            count=test_img.shape[0],
            height=test_img.shape[1],
            width=test_img.shape[2],
            dtype=test_img.dtype,
            crs=src.crs,
            transform=transform,
        ) as dst:
            dst.write(test_img)


def get_args():

    parser = argparse.ArgumentParser(description="Geo-Reference generated images")
    parser.add_argument(
        "--generated_dir",
        "-t",
        type=str,
        help="Path to directory containing generated image patches",
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        help="Path to directory containing input OSM rasters",
    )
    parser.add_argument("--output_dir", "-o", type=str, help="Path to output directory")

    args = parser.parse_args()
    return args


def main():

    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)

    processed, missing = 0, []
    for name in sorted(os.listdir(args.generated_dir)):

        # Retrieve real image name
        match = GENERATED_SUFFIX_RE.match(name)
        if match is None:
            continue
        ref_name = match.group("ref") + ".tif"

        ref_path = os.path.join(args.input_dir, ref_name)
        if os.path.isfile(ref_path):
            geoloc_generated_images(
                ref_path, os.path.join(args.generated_dir, name), args.output_dir
            )
            processed += 1
        else:
            missing.append(ref_name)

    print(f"Georeferenced {processed} image(s) into {args.output_dir}")
    if missing:
        print(f"WARNING: no reference raster found in {args.input_dir} for {len(missing)} image(s), e.g. {missing[0]}")
    if processed == 0:
        raise SystemExit("No generated image was georeferenced: check --generated_dir and --input_dir")


if __name__ == "__main__":
    main()
