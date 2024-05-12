import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from pathlib import Path
import torch
import nibabel as nib
import torch.nn.functional as F
import torch.nn as nn
from typing import Literal
from monai.networks.nets import UNet
# Structural Similarity Loss
from monai.losses import SSIMLoss
from monai.networks.layers import Norm
from monai.data import CacheDataset, Dataset, DataLoader, list_data_collate, NibabelReader
from monai.data import MetaTensor
from monai.transforms import MapTransform


def training_data_generator_pt(seismic, axis: Literal['i_line', 'x_line', None]=None, percentile: int=25):
    # check parameters
    assert isinstance(seismic, MetaTensor) and len(seismic.shape)==4, 'seismic must be 4D MetaTensor'
    assert axis in ['i_line', 'x_line', None], 'axis must be one of: i_line, x_line or None'
    assert type(percentile) is int and 0<percentile<100, 'percentile must be an integer between 0 and 100'

    # if axis is None get random choice
    if axis is None:
        axis = np.random.choice(['i_line', 'x_line'], 1)[0]

    # crop subset
    if axis == 'i_line':
        sample_size = np.round(seismic.shape[1]*(percentile/100)).astype('int')
        sample_start = np.random.choice(range(seismic.shape[1]-sample_size), 1)[0]
        sample_end = sample_start+sample_size

        target_mask = np.zeros(seismic.shape).astype('bool')
        target_mask[:, sample_start:sample_end, :, :] = True

        target = seismic[:, sample_start:sample_end, :, :].clone()
        seismic = MetaTensor(np.where(target_mask, np.nan, seismic))

    else:
        sample_size = np.round(seismic.shape[2]*(percentile/100)).astype('int')
        sample_start = np.random.choice(range(seismic.shape[2]-sample_size), 1)[0]
        sample_end = sample_start+sample_size

        target_mask = np.zeros(seismic.shape).astype('bool')
        target_mask[:, :, sample_start:sample_end, :] = True
        target = seismic[:, :, sample_start:sample_end, :].clone()
        seismic = MetaTensor(np.where(target_mask, np.nan, seismic))

    return seismic, torch.Tensor(target_mask).type(torch.bool)
