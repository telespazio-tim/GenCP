import os.path
import csv
import math

from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image

import torch


#  Load geographic coordinates from the precomputed CSV file.
def load_geo_coordinates(csv_path):
    coordinates = {}

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)

        required = {'filename', 'xs', 'ys', 'zs'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Coordinate CSV must contain filename,xs,ys,zs')
        for row in reader:
            if row['filename'] in coordinates:
                raise ValueError('Duplicate coordinate filename: ' + row['filename'])
            coordinates[row['filename']] = (
                float(row['xs']),
                float(row['ys']),
                float(row['zs'])
            )

    if not all(math.isfinite(v) for values in coordinates.values() for v in values):
        raise ValueError('Coordinates must be finite numbers')
    return coordinates


class AlignedDataset(BaseDataset):

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        #  Load geographic coordinates once during dataset initialization.
        self.geo_coordinates = load_geo_coordinates(
            opt.coord_csv or os.path.join(opt.dataroot, 'geo_coordinates.csv')
        )

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):

        ### input A
        A_path = self.A_paths[index]
        A = Image.open(A_path)

        params = get_params(self.opt, A.size)

        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(
                self.opt,
                params,
                method=Image.NEAREST,
                normalize=False
            )
            A_tensor = transform_A(A) * 255.0

        #  Keep geographic coordinates as global metadata instead of expanding them into image channels.
        filename = os.path.basename(A_path)

        if filename not in self.geo_coordinates:
            raise KeyError(
                'Coordinate is not found in CSV file: %s' % filename
            )

        xs, ys, zs = self.geo_coordinates[filename]
        coord_tensor = torch.tensor(
            [xs, ys, zs],
            dtype=torch.float32
        )

        B_tensor = inst_tensor = feat_tensor = 0

        ### input B
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)
            B_tensor = transform_B(B)

        ### if using instance maps
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))

        input_dict = {
            'label': A_tensor,
            'coord': coord_tensor,
            'inst': inst_tensor,
            'image': B_tensor,
            'feat': feat_tensor,
            'path': A_path
        }

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'