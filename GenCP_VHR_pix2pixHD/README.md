# pix2pixHD for OSM-to-satellite image translation

This project adapts [NVIDIA pix2pixHD](https://github.com/NVIDIA/pix2pixHD) for paired translation from RGB OSM map images to RGB satellite images. It provides a coordinate-conditioned bottleneck model, with notebooks for training, inference and geospatial post-processing.

The implementation builds on [Wang et al., CVPR 2018](https://arxiv.org/abs/1711.11585).

## Models and datasets

`V2` denotes the dataset version used for the reference experiment and retained in the experiment and checkpoint names. The complete collection contains 59,302 images using the new OSM mask style, including both training and test samples.

| Experiment directory | Dataset | Mask style | Model mode |
|---|---|---|---|
| `Model_V2_dataset_global_D3_allloss_aug_batch_coord_bottleneck` | V2 | New | `bottleneck` |

New-style masks were generated using the [VHR dataset preparation code][data-preparation]. Dataset sizes refer to the complete collections, not individual train/test splits.

### Coordinate conditioning

This implementation uses geographic XYZ conditioning at the generator bottleneck. An MLP maps unit-sphere XYZ metadata to an embedding that is added to the encoder output before the residual blocks. The final MLP layer is initialized to zero.

Input and output images have three RGB channels. Coordinates are supplied separately from the image channels; the discriminator does not receive them directly. Both `global` and `local` generators are supported.

## Installation

The working environment uses **Python 3.10.20**, **PyTorch 2.5.1+cu121**, and **torchvision 0.20.1+cu121**. The following commands recreate the main pix2pixHD environment in a new Conda environment. If an existing environment is already working correctly, it is recommended to keep it unchanged.

### Create the pix2pixHD environment

```bat
conda create -n pix2pixhd python=3.10.20 -y

conda activate pix2pixhd

python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

python -m pip install "setuptools==81.0.0" "wheel==0.46.3"

python -m pip install -r requirements.txt

python -m ipykernel install --user --name pix2pixhd --display-name "Python (pix2pixhd)"
```

After installation, verify the PyTorch version, CUDA version, and GPU availability:

```bat
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

### Python dependencies

The remaining Python dependencies required by the project are listed in:

```text
requirements.txt
```

They are installed with:

```bat
python -m pip install -r requirements.txt
```

PyTorch and torchvision are installed separately so that the required CUDA 12.1 builds are selected explicitly.

GDAL is not included in `requirements.txt`. It is required for coordinate generation, georeferencing, and merging, and can be installed in a separate environment.

For additional installation instructions and usage examples, see the [official NVIDIA pix2pixHD repository](https://github.com/NVIDIA/pix2pixHD).

## Data preparation

Download the dataset from [test dataset](https://mega.nz/folder/aXohTBpB#W-MPbpq_E6RoCnHqz24-rw) and place the image folders in the project's `datasets/` directory. If `geo_coordinates.csv` is not included, generate it as described below.

Set `DATAROOT` to the dataset directory containing:

| Path | Contents |
|---|---|
| `train_A/` | RGB OSM training inputs |
| `train_B/` | Corresponding RGB satellite targets |
| `test_A/` | RGB OSM test inputs |
| `test_B/` | Corresponding reference satellite images; optional for ordinary inference |
| `geo_coordinates.csv` | Required geographic coordinate metadata |

A/B training files must correspond in sorted order and cover the same geographic footprint. Ordinary inference uses `test_A` and the coordinate CSV. Georeferencing copies the CRS and affine transform from the matching OSM GeoTIFFs in `test_A`.

By default, `DATAROOT` points to the project's `datasets/` directory. It can be changed to another dataset location.

For the model, run `generate_coordinate_csv.ipynb` in a GDAL-enabled environment. It reads the geographic centre of each input GeoTIFF in `train_A` and `test_A`, transforms it to EPSG:4326 and converts longitude/latitude to **unit-sphere XYZ coordinates**. These values describe location, not elevation or metric XYZ coordinates. For a test-only dataset, set `FOLDERS = ["test_A"]`.

The model reads `filename,xs,ys,zs` from `DATAROOT/geo_coordinates.csv`. Filenames must be unique across the CSV. Preserve the coordinate convention used during training and provide coordinate metadata for every input image.

## Training and testing

1. Open `notebooks/train.ipynb` or `notebooks/test.ipynb`.
2. Set `DATAROOT`, `NAME` and checkpoint paths.
3. Edit numerical/string arguments in `settings`; add Boolean switches to `flags`.
4. Run the command-building cell, review the command, then run the execution cell.

Testing uses **batch size 1**, `eval()` and `torch.no_grad()`. Training save/display intervals count processed images. `continue_train` does not restore the complete optimizer/RNG state and is not equivalent to exact uninterrupted training.

## Model weights

Download the pretrained generator weights using the link below.

| Experiment | Weights |
|---|---|
| V2, coordinate bottleneck | [Model_V2_dataset_global_D3_allloss_aug_batch_coord_bottleneck](https://mega.nz/folder/SPxCGBoT#245fLh4Ot3JNWTp9aeRvDg) |

Place `200_net_G.pth` in `checkpoints/Model_V2_dataset_global_D3_allloss_aug_batch_coord_bottleneck/` and set `which_epoch` to `"200"`.

## Georeferencing and merge

Use the separate **Georeferencing** and **Merge** sections in `notebooks/test.ipynb` after inference. When switching to a GDAL kernel, rerun the setup and settings cells to restore paths, then run post-processing without repeating inference.

Georeferencing copies the CRS and affine transform from matching test_B GeoTIFFs when available; otherwise, it uses the OSM GeoTIFFs in test_A. It assumes that predictions and reference tiles cover the same geographic footprint and does not perform image registration.

Outputs are saved under `results/<NAME>/<phase>_<epoch>/`:

- `images/` and `index.html`: initial JPG predictions and visualizations.
- `geotiff/input_label/` and `geotiff/synthesized_image/`: georeferenced tiles.
- `geotiff/input_label_merged/` and `geotiff/synthesized_image_merged/`: mosaics.

Geospatial outputs are **GeoTIFF (`.tif`)**. Converting the initial JPG predictions to GeoTIFF does not recover JPEG compression losses. Existing geospatial outputs are retained unless overwrite is enabled.

## Project files

Model implementations are in `models/`, dataset loaders in `data/`, command-line options in `options/`, and post-processing utilities in `util/`. The notebooks call the shared `train.py` and `test.py` entry points.

Keep weights and generated outputs outside source control. Empty `checkpoints/` and `results/` directories can be represented by `.gitkeep` files with the corresponding `.gitignore` exceptions.

## License and citation

Retain the complete [LICENSE.txt](LICENSE.txt), including the original NVIDIA and pytorch-CycleGAN-and-pix2pix notices. See the [upstream license](https://github.com/NVIDIA/pix2pixHD/blob/master/LICENSE.txt) for the conditions governing use, modification and redistribution. Dataset and imagery rights are separate from the code license.

```bibtex
@inproceedings{wang2018pix2pixHD,
  title={High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs},
  author={Wang, Ting-Chun and Liu, Ming-Yu and Zhu, Jun-Yan and Tao, Andrew and Kautz, Jan and Catanzaro, Bryan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
}
```

[data-preparation]: ../GenCP_data_preparation/VHR/README.md
