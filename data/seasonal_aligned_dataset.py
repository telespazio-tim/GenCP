import os
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset, IMG_EXTENSIONS
from PIL import Image


SEASONS = ('winter', 'summer')
SEASON_VALUES = {'winter': -1.0, 'summer': 1.0}


def truncate_balanced(samples, max_size, season_of=lambda sample: sample[-1]):
    """Keep at most ``max_size`` samples, taken alternately from each season.

    A plain ``samples[:max_size]`` on a list sorted by season would keep only
    winter samples when ``max_size`` is small.
    """
    by_season = {}
    for sample in samples:
        by_season.setdefault(season_of(sample), []).append(sample)
    kept = []
    queues = [list(reversed(v)) for v in by_season.values()]
    while len(kept) < max_size and any(queues):
        for queue in queues:
            if queue and len(kept) < max_size:
                kept.append(queue.pop())
    return kept


def _is_image(fname):
    return any(fname.endswith(ext) for ext in IMG_EXTENSIONS)


class SeasonalAlignedDataset(BaseDataset):
    """Paired dataset with a seasonal conditioning channel.

    Expected layout:
        {dataroot}/{phase}/summer/*.{png,jpg,...}
        {dataroot}/{phase}/winter/*.{png,jpg,...}

    Each file is a side-by-side A|B pair (map|photo) — same convention as
    AlignedDataset. The season is inferred from the subfolder and appended
    to A as a constant 4th channel: winter = -1, summer = +1.
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
            season_dir = os.path.join(phase_dir, season)
            if not os.path.isdir(season_dir):
                continue
            for root, _, fnames in sorted(os.walk(season_dir)):
                for fname in sorted(fnames):
                    if _is_image(fname):
                        self.samples.append((os.path.join(root, fname), season))
        if not self.samples:
            raise RuntimeError(
                "No images found under %s/{summer,winter}/" % phase_dir)
        if opt.max_dataset_size < len(self.samples):
            self.samples = truncate_balanced(self.samples, int(opt.max_dataset_size))

        assert self.opt.load_size >= self.opt.crop_size
        # A carries an extra season channel; input_nc already accounts for it.
        self.rgb_input_nc = (self.opt.output_nc if self.opt.direction == 'BtoA'
                             else self.opt.input_nc) - 1
        self.output_nc = (self.opt.input_nc - 1 if self.opt.direction == 'BtoA'
                          else self.opt.output_nc)

    def __getitem__(self, index):
        AB_path, season = self.samples[index]
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        w2 = w // 2
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

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

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path,
                'season': season}

    def __len__(self):
        return len(self.samples)
