"""
batch_formatter.py – Batch OSM Formatting Runner

This script performs batch formatting of downloaded OSM shapefile outputs.
It scans the main input directory, detects each subfolder, and calls
osm_formatter_engine.py separately for every subfolder.

The actual formatting, layer conversion, attribute editing, and output generation
are handled by osm_formatter_engine.py. This script only manages batch execution,
folder iteration, input/output path creation, and error reporting.

Input parameters:
--input / -i:
Root directory containing the raw OSM shapefile folders.
Each subfolder is processed separately.

--output / -o:
    Root directory where the formatted OSM outputs will be saved.
    A corresponding output subfolder is created for each input subfolder.

Outputs:
- Each input subfolder is processed by osm_formatter_engine.py.
- Formatted outputs are saved into the corresponding output subfolder.
- The processing status of each folder is printed to the console.
- If an error occurs in one folder, the error is reported and the script
continues with the next folder.

"""



import os
import sys
import subprocess
import argparse


def batch_run(input_root, output_root):
    if not os.path.exists(input_root):
        print("Error: Main entry folder not found!")
        return

    subfolders = [
        f for f in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, f))
    ]

    print(f"--- Total of {len(subfolders)} folders will be processed ---")

    engine_script = os.path.join(os.path.dirname(__file__), "osm_formatter_engine.py")

    for sub in subfolders:
        in_dir  = os.path.join(input_root,  sub)
        out_dir = os.path.join(output_root, sub)

        print(f"\n Processing: {sub}")

        try:
            subprocess.run(
                [
                    sys.executable, engine_script,
                    "--input",  in_dir,
                    "--output", out_dir,
                ],
                check=True,
            )
            print(f"[SUCCESSFUL] {sub}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {sub} failed during processing: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch OSM format conversion: processes all shapefiles in subfolders."
    )
    parser.add_argument("--input",  "-i", required=True, help="Root directory containing the raw shapefile folders")
    parser.add_argument("--output", "-o", required=True, help="Root directory where the processed outputs will be saved")
    args = parser.parse_args()

    batch_run(args.input, args.output)