import os, sys
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))

from . import detectors
from . import hand_datasets
from . import body_datasets
from . import openpose_dataset


def build_dataloader(data_dir, dataset_type = 'body', batch_size=1, split = None, device = 'cpu', logger = None):
    if dataset_type == 'body':
        dataset = body_datasets.TestData(testpath = data_dir, device = device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=1,
                                pin_memory=True,
                                drop_last =False)
        return dataset, dataloader
    
    elif dataset_type == 'openpose':
        dataset = openpose_dataset.OpenPoseDataset(data_dir = data_dir, split = split, logger = logger)
        if split == 'train':
            drop_last = True
            shuffle = True
        else:
            drop_last = False
            shuffle = False
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                num_workers=4,
                                pin_memory=False,
                                drop_last =drop_last)
        return dataset, dataloader
    
    else:
        logger.info(f'invalid dataset type: {dataset_type}')
        exit()