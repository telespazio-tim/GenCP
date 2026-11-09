"""Load the coordinate-conditioned dataset while preserving the loader API."""
from data.aligned_dataset_bottleneck import AlignedDataset as BottleneckDataset

def AlignedDatasetForMode(opt):
    dataset = BottleneckDataset()
    dataset.initialize(opt)
    if not dataset.A_paths:
        raise ValueError('No input images found: ' + dataset.dir_A)
    if opt.isTrain or getattr(opt, 'use_encoded_image', False):
        if len(dataset.A_paths) != len(dataset.B_paths):
            raise ValueError('A and B image counts differ. Check paired data.')
    if len(dataset) == 0:
        raise ValueError('Dataset has fewer images than batchSize; lower batchSize.')
    return dataset