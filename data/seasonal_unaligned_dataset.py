import os
import random
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from data.seasonal_aligned_dataset import SEASONS, SEASON_VALUES, truncate_balanced
from PIL import Image


class SeasonalUnalignedDataset(BaseDataset):
    """Unpaired dataset with a seasonal conditioning channel.

    Expected layout:
        {dataroot}/{phase}/A_summer/*.{png,jpg,...}
        {dataroot}/{phase}/A_winter/*.{png,jpg,...}
        {dataroot}/{phase}/B_summer/*.{png,jpg,...}
        {dataroot}/{phase}/B_winter/*.{png,jpg,...}

    Each A sample carries its season as a constant 4th channel
    (winter = -1, summer = +1). B is drawn from the same season as A so
    that adversarial and cycle losses stay seasonally consistent.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(input_nc=4)
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        phase_dir = os.path.join(opt.dataroot, opt.phase)

        self.A_paths_by_season = {}
        self.B_paths_by_season = {}
        self.A_paths = []
        for season in SEASONS:
            dir_A = os.path.join(phase_dir, 'A_' + season)
            dir_B = os.path.join(phase_dir, 'B_' + season)
            if not (os.path.isdir(dir_A) and os.path.isdir(dir_B)):
                continue
            a_paths = sorted(make_dataset(dir_A, opt.max_dataset_size))
            b_paths = sorted(make_dataset(dir_B, opt.max_dataset_size))
            if not a_paths or not b_paths:
                continue
            self.A_paths_by_season[season] = a_paths
            self.B_paths_by_season[season] = b_paths
            self.A_paths.extend((p, season) for p in a_paths)

        if not self.A_paths:
            raise RuntimeError(
                "No images found under %s/{A_summer,A_winter,B_summer,B_winter}/"
                % phase_dir)
        if opt.max_dataset_size < len(self.A_paths):
            self.A_paths = truncate_balanced(self.A_paths, int(opt.max_dataset_size))

        btoA = self.opt.direction == 'BtoA'
        # input_nc already accounts for the season channel; subtract it for the
        # RGB-only transform.
        input_nc = (self.opt.output_nc if btoA else self.opt.input_nc) - 1
        output_nc = self.opt.input_nc - 1 if btoA else self.opt.output_nc
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        A_path, season = self.A_paths[index % len(self.A_paths)]
        B_paths = self.B_paths_by_season[season]
        if self.opt.serial_batches:
            index_B = index % len(B_paths)
        else:
            index_B = random.randint(0, len(B_paths) - 1)
        B_path = B_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        season_value = SEASON_VALUES[season]
        season_channel = torch.full((1, A.shape[1], A.shape[2]),
                                    season_value, dtype=A.dtype)
        A = torch.cat([A, season_channel], dim=0)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path,
                'season': season}

    def __len__(self):
        return len(self.A_paths)
