import os
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import is_image_file
from data.seasonal_aligned_dataset import SEASONS, SEASON_VALUES, truncate_balanced
from PIL import Image


class SeasonalPairedFoldersDataset(BaseDataset):
    """Paired dataset with a seasonal conditioning channel, split into folders.

    Expected layout:
        {dataroot}/{phase}/A_summer/*.{png,jpg,...}
        {dataroot}/{phase}/B_summer/*.{png,jpg,...}
        {dataroot}/{phase}/A_winter/*.{png,jpg,...}
        {dataroot}/{phase}/B_winter/*.{png,jpg,...}

    A and B are paired by filename within each season. Files present in only
    one of A_<season>/B_<season> are skipped. The season is appended to A as
    a constant 4th channel: winter = -1, summer = +1.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(input_nc=4)
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        phase_dir = os.path.join(opt.dataroot, opt.phase)

        self.samples = []
        for season in SEASONS:
            dir_A = os.path.join(phase_dir, 'A_' + season)
            dir_B = os.path.join(phase_dir, 'B_' + season)
            if not (os.path.isdir(dir_A) and os.path.isdir(dir_B)):
                continue
            a_files = {f for f in os.listdir(dir_A) if is_image_file(f)}
            b_files = {f for f in os.listdir(dir_B) if is_image_file(f)}
            common = sorted(a_files & b_files)
            only_a = a_files - b_files
            only_b = b_files - a_files
            if only_a or only_b:
                print("[seasonal_paired_folders] %s: skipping %d A-only and "
                      "%d B-only files" % (season, len(only_a), len(only_b)))
            for fname in common:
                self.samples.append((os.path.join(dir_A, fname),
                                     os.path.join(dir_B, fname),
                                     season))

        if not self.samples:
            raise RuntimeError(
                "No paired images found under %s/{A_summer,B_summer,"
                "A_winter,B_winter}/" % phase_dir)
        if opt.max_dataset_size < len(self.samples):
            self.samples = truncate_balanced(self.samples, int(opt.max_dataset_size))

        assert self.opt.load_size >= self.opt.crop_size
        self.rgb_input_nc = (self.opt.output_nc if self.opt.direction == 'BtoA'
                             else self.opt.input_nc) - 1
        self.output_nc = (self.opt.input_nc - 1 if self.opt.direction == 'BtoA'
                          else self.opt.output_nc)

    def __getitem__(self, index):
        A_path, B_path, season = self.samples[index]
        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')

        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params,
                                    grayscale=(self.rgb_input_nc == 1))
        B_transform = get_transform(self.opt, transform_params,
                                    grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        season_value = SEASON_VALUES[season]
        season_channel = torch.full((1, A.shape[1], A.shape[2]),
                                    season_value, dtype=A.dtype)
        A = torch.cat([A, season_channel], dim=0)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path,
                'season': season}

    def __len__(self):
        return len(self.samples)
