import os

import numpy as np
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
from data.dem_dataset import find_dem_dirs
from data.image_folder import make_dataset
from PIL import Image


class DemSingleDataset(BaseDataset):
    """
    Pix2Pix DEM Dataset where input A = RGB + DEM (4 canaux)
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        root, self.dir_A_dem = find_dem_dirs(opt)
        self.dir_A_rgb = os.path.join(root, 'A')

        self.A_rgb_paths = sorted(make_dataset(self.dir_A_rgb, opt.max_dataset_size))
        self.A_dem_paths = sorted(make_dataset(self.dir_A_dem, opt.max_dataset_size))

        # Check if folders have the same length
        assert len(self.A_rgb_paths) == len(self.A_dem_paths), \
            f"Folders should have the same length" \
            f"A: {len(self.A_rgb_paths)}, DEM: {len(self.A_dem_paths)}"

        # Check file's names consistency
        for i in range(len(self.A_rgb_paths)):
            a_name = os.path.basename(self.A_rgb_paths[i])
            dem_name = os.path.basename(self.A_dem_paths[i])
            
            a_base = os.path.splitext(a_name)[0]
            dem_base = os.path.splitext(dem_name)[0]
            
            if a_base != dem_base :
                raise ValueError(
                    f"Files not matching index {i}:\n"
                    f"  A: {a_name}\n"
                    f"  DEM: {dem_name}\n"
                )

    def __getitem__(self, index):
        A_rgb_path = self.A_rgb_paths[index]
        A_dem_path = self.A_dem_paths[index]
        
        A_rgb = Image.open(A_rgb_path).convert('RGB')
        A_dem = Image.open(A_dem_path).convert('L')  # grayscale conversion for DEM files

        # Same crop / flip for every layer, otherwise RGB, DEM and B are misaligned
        transform_params = get_params(self.opt, A_rgb.size)
        transform_A_rgb = get_transform(self.opt, transform_params, grayscale=False)
        transform_A_dem = get_transform(self.opt, transform_params, grayscale=True)

        A_rgb_tensor = transform_A_rgb(A_rgb)  # [3, H, W]
        A_dem_tensor = transform_A_dem(A_dem)  # [1, H, W]

        # Concatenate tensors
        A = torch.cat([A_rgb_tensor, A_dem_tensor], dim=0)  # [4, H, W]

        return {
            'A': A, 
            'A_paths': A_rgb_path,
        }

    def __len__(self):
        return len(self.A_rgb_paths)
