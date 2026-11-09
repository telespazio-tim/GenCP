import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from data.seasonal_aligned_dataset import SEASONS, SEASON_VALUES
from PIL import Image


class SeasonalSingleDataset(BaseDataset):
    """Unpaired inference dataset with a seasonal conditioning channel.

    Loads map rasters from --dataroot and appends a constant season channel
    (winter = -1, summer = +1) so the input has the 4 channels expected by
    a model trained with seasonal_aligned. The season comes from --season.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(input_nc=4)
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        if getattr(opt, 'season', None) not in SEASONS:
            raise ValueError(
                "--season must be one of %s for seasonal_single dataset"
                % (SEASONS,))
        self.season = opt.season
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        rgb_input_nc = (self.opt.output_nc if self.opt.direction == 'BtoA'
                        else self.opt.input_nc) - 1
        self.transform = get_transform(opt, grayscale=(rgb_input_nc == 1))

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)

        season_value = SEASON_VALUES[self.season]
        season_channel = torch.full((1, A.shape[1], A.shape[2]),
                                    season_value, dtype=A.dtype)
        A = torch.cat([A, season_channel], dim=0)

        return {'A': A, 'A_paths': A_path, 'season': self.season}

    def __len__(self):
        return len(self.A_paths)
