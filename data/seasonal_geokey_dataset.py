"""
seasonal_geokey_dataset.py
Dataset class for pix2pix with cross-season geometric stability loss.

Differs from SeasonalPairedFoldersDataset:
- Indexes files by "geographic key": the file stem without a trailing
  _summer / _winter (e.g. T31TGH_tile_0001_0001 or a GRI chip name)
- Keeps only keys present in all 4 directories (A_summer, B_summer, A_winter, B_winter)
- Returns at each call the 4 tensors (A_summer, B_summer, A_winter, B_winter)
  of the same geographic area for the summer+winter double forward pass.

Place this file in data/seasonal_geokey_dataset.py
"""
import os
import re
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import is_image_file
from data.seasonal_aligned_dataset import SEASONS, SEASON_VALUES
from PIL import Image


# Season suffix removed from the stem to get the geographic key
# ex: "T31TGH_tile_0001_0001_summer.png" -> "T31TGH_tile_0001_0001"
# ex: "31TEJ_0451_00.tif"                -> "31TEJ_0451_00"
SEASON_SUFFIX_PATTERN = re.compile(r"_(?:summer|winter)$", re.IGNORECASE)


def extract_geo_key(filename):
    """Extract the geographic key: the stem without a trailing season suffix."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    return SEASON_SUFFIX_PATTERN.sub("", stem)


class SeasonalGeoKeyDataset(BaseDataset):
    """Paired dataset where each sample is a (summer+winter) quadruplet of the same area.

    Expected layout (identical to SeasonalPairedFoldersDataset):
        {dataroot}/{phase}/A_summer/{geokey}[_summer].png
        {dataroot}/{phase}/B_summer/{geokey}[_summer].png
        {dataroot}/{phase}/A_winter/{geokey}[_winter].png
        {dataroot}/{phase}/B_winter/{geokey}[_winter].png

    A key is kept only if it is present in all 4 directories.

    A and B receive the season channel (4 input channels for A).
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(input_nc=4)
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        phase_dir = os.path.join(opt.dataroot, opt.phase)

        # Index each directory by geographic key
        def index_dir(d):
            if not os.path.isdir(d):
                return {}
            result = {}
            for f in os.listdir(d):
                if not is_image_file(f):
                    continue
                result[extract_geo_key(f)] = os.path.join(d, f)
            return result

        idx_A_s = index_dir(os.path.join(phase_dir, "A_summer"))
        idx_B_s = index_dir(os.path.join(phase_dir, "B_summer"))
        idx_A_w = index_dir(os.path.join(phase_dir, "A_winter"))
        idx_B_w = index_dir(os.path.join(phase_dir, "B_winter"))

        print("[seasonal_geokey] Indexed:")
        print(f"  A_summer: {len(idx_A_s)} files")
        print(f"  B_summer: {len(idx_B_s)} files")
        print(f"  A_winter: {len(idx_A_w)} files")
        print(f"  B_winter: {len(idx_B_w)} files")

        # Keys present in all 4 directories
        common_keys = sorted(
            set(idx_A_s) & set(idx_B_s) & set(idx_A_w) & set(idx_B_w)
        )
        print(f"  -> {len(common_keys)} keys present in all 4 directories")

        self.samples = []
        for key in common_keys:
            self.samples.append({
                "A_summer": idx_A_s[key],
                "B_summer": idx_B_s[key],
                "A_winter": idx_A_w[key],
                "B_winter": idx_B_w[key],
                "geokey": key,
            })

        if not self.samples:
            raise RuntimeError(
                f"No quadruplets found under {phase_dir}. "
                "Check that A_summer/B_summer/A_winter/B_winter all contain "
                "files with the same names (a trailing _summer / _winter is ignored)."
            )

        if opt.max_dataset_size < len(self.samples):
            self.samples = self.samples[:int(opt.max_dataset_size)]

        assert self.opt.load_size >= self.opt.crop_size
        self.rgb_input_nc = (
            self.opt.output_nc if self.opt.direction == "BtoA"
            else self.opt.input_nc
        ) - 1
        self.output_nc = (
            self.opt.input_nc - 1 if self.opt.direction == "BtoA"
            else self.opt.output_nc
        )

    def _load_and_transform_pair(self, A_path, B_path, season, transform_params):
        """Load A and B with the season channel, using the same transform_params
        to ensure a consistent A/B alignment (identical crops and flips)."""
        A = Image.open(A_path).convert("RGB")
        B = Image.open(B_path).convert("RGB")

        A_transform = get_transform(
            self.opt, transform_params, grayscale=(self.rgb_input_nc == 1)
        )
        B_transform = get_transform(
            self.opt, transform_params, grayscale=(self.output_nc == 1)
        )
        A = A_transform(A)
        B = B_transform(B)

        # Append the season channel
        season_value = SEASON_VALUES[season]
        season_channel = torch.full(
            (1, A.shape[1], A.shape[2]), season_value, dtype=A.dtype
        )
        A = torch.cat([A, season_channel], dim=0)
        return A, B

    def __getitem__(self, index):
        sample = self.samples[index]

        # Compute transform_params ONCE for consistency between summer and winter
        # (same crop, same flip -> the geographic area stays aligned)
        ref_image = Image.open(sample["A_summer"]).convert("RGB")
        transform_params = get_params(self.opt, ref_image.size)

        A_summer, B_summer = self._load_and_transform_pair(
            sample["A_summer"], sample["B_summer"], "summer", transform_params
        )
        A_winter, B_winter = self._load_and_transform_pair(
            sample["A_winter"], sample["B_winter"], "winter", transform_params
        )

        return {
            "A_summer": A_summer, "B_summer": B_summer,
            "A_winter": A_winter, "B_winter": B_winter,
            "A_summer_paths": sample["A_summer"],
            "B_summer_paths": sample["B_summer"],
            "A_winter_paths": sample["A_winter"],
            "B_winter_paths": sample["B_winter"],
            "geokey": sample["geokey"],
        }

    def __len__(self):
        return len(self.samples)