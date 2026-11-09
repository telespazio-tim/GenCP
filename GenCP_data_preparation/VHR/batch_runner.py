#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
batch_runner.py – Batch OSM Data Download Runner

This script is used to run OSM data download operations in batch mode.
It scans the input folder for raster or vector files and calls osm_processor.py
separately for each input file. The actual OSM data download, bounding box
extraction, buffer operation, layer filtering, and output generation are handled
by osm_processor.py.

Input parameters:
--input:
Folder containing the raster/vector files to be processed.

--output:
    Main output folder where OSM results will be saved.

--ext:
    File extensions to be processed. Default: tif.
    Example: --ext tif shp geojson

--recursive:
    If used, subfolders inside the input folder are also scanned.

--buffer:
    Buffer distance around the input boundary or bounding box, in meters.

--layers:
    OSM layers to be downloaded.
    Example: buildings roads waterways

--stop_on_error:
    Stops the batch process if an error occurs.

Outputs:
- A separate output folder is created for each input file.
- OSM outputs are saved by osm_processor.py into the related folder.
- The number of successful and failed processes is printed at the end.

Usage examples:
python batch_runner.py --input "C:...\rasters" --output "C:...\out"

python batch_runner.py --input "C:\...\vectors" --output "C:\...\out" --ext shp

python batch_runner.py --input "C:\...\mixed" --output "C:\...\out" --ext tif shp geojson

python batch_runner.py --input "C:\...\main" --output "C:\...\out" --recursive

python batch_runner.py --input "C:\...\rasters" --output "C:\...\out" --buffer 100 --layers buildings roads waterways

python batch_runner.py --input "C:\...\rasters" --output "C:\...\out" --stop_on_error


"""

import os
import sys
import glob
import subprocess
import argparse

# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Batch OSM data download for raster/vector files in a folder."
)
parser.add_argument(
    "--input",  "-i", required=True,
    help="Folder containing the input files"
)
parser.add_argument(
    "--output", "-o", required=True,
    help="Root folder where the outputs will be saved"
)
parser.add_argument(
    "--ext", nargs="*", default=["tif"],
    help=(
        "File extensions to be processed (without a dot, default: tif)."
        "e.g.: --ext tif shp geojson"
    )
)
parser.add_argument(
    "--recursive", action="store_true",
    help="Scan subfolders as well"
)
parser.add_argument(
    "--buffer", type=float, default=0,
    help="BBox buffer in meters (passed to osm_processor, default: 0)"
)
parser.add_argument(
    "--layers", nargs="*", default=None,
    help="Short names of OSM layers to be processed (passed to osm_processor, default: all)"
)
parser.add_argument(
    "--stop_on_error", action="store_true",
    help="Stop the entire process if an error occurs in any file"
)

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Create the file list
# ---------------------------------------------------------------------------

def collect_files(input_root, extensions, recursive):
    files = []
    exts  = {f".{e.lstrip('.')}" .lower() for e in extensions}

    if recursive:
        for root, dirs, filenames in os.walk(input_root):
            for fname in filenames:
                if os.path.splitext(fname)[1].lower() in exts:
                    files.append(os.path.join(root, fname))
    else:
        for fname in os.listdir(input_root):
            if os.path.splitext(fname)[1].lower() in exts:
                files.append(os.path.join(input_root, fname))

    return sorted(files)

# ---------------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------------

def run_batch(input_root, output_root, extensions, recursive,
              buffer_meters, layer_filter, stop_on_error):

    if not os.path.exists(input_root):
        print(f"[ERROR] Source folder not found: {input_root}")
        sys.exit(1)

    file_list = collect_files(input_root, extensions, recursive)

    if not file_list:
        print(f"[WARNING] No files with the specified extensions ({extensions}) were found: {input_root}")
        sys.exit(0)

    print(f"Input      : {input_root}")
    print(f"Output      : {output_root}")
    print(f"Extensions   : {extensions}")
    print(f"Recursive   : {recursive}")
    print(f"Buffer      : {buffer_meters} m")
    print(f"Layer   : {layer_filter or 'all'}")
    print(f"Number of files: {len(file_list)}")
    print("-" * 55)

    osm_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "osm_processor.py")
    if not os.path.exists(osm_script):
        print(f"[ERROR] osm_processor.py not found:: {osm_script}")
        sys.exit(1)

    success_count = 0
    fail_count    = 0

    for i, input_path in enumerate(file_list, 1):
        filename    = os.path.basename(input_path)
        folder_name = os.path.splitext(filename)[0]

        # Preserve the subfolder structure in recursive mode
        if recursive:
            rel_path    = os.path.relpath(os.path.dirname(input_path), input_root)
            target_dir  = os.path.join(output_root, rel_path, folder_name)
        else:
            target_dir  = os.path.join(output_root, folder_name)

        print(f"\n[{i}/{len(file_list)}] {filename}")
        print(f"  Output: {target_dir}")

        cmd = [
            sys.executable, osm_script,
            "--input",  input_path,
            "--output", target_dir,
            "--buffer", str(buffer_meters),
        ]
        if layer_filter:
            cmd += ["--layers"] + layer_filter

        try:
            subprocess.run(cmd, check=True)
            print("  [SUCCESSFUL]")
            success_count += 1
        except subprocess.CalledProcessError as e:
            fail_count += 1
            print(f"  [ERROR] Exit code: {e.returncode}")
            if stop_on_error:
                print("[STOPPED] --stop_on_error is active, terminating the process.")
                break

    print("\n" + "=" * 55)
    print(f"COMPLETED  |  Successful: {success_count}  |  Failed: {fail_count}")
    print(f"Output      : {output_root}")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_batch(
        input_root    = args.input,
        output_root   = args.output,
        extensions    = args.ext,
        recursive     = args.recursive,
        buffer_meters = args.buffer,
        layer_filter  = args.layers,
        stop_on_error = args.stop_on_error,
    )