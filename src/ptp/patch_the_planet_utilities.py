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

DIMS = ['x', 'y', 'z']

def visualize_slices(nib_img, x, y, z, title):
    plt.figure(figsize=(12, 4), constrained_layout=True)
    plt.suptitle(title)
    slices = [np.s_[x, :, :], np.s_[:, y, :], np.s_[:, :, z]]
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(nib_img.get_fdata()[slices[i]], cmap='Greys')
        plt.title(f'Slice along {DIMS[i]}')


def visualize_volumes(data_dict, idx, keys, labels, is_color_channel=False):
    nonzero_mask = torch.nonzero(data_dict['mask'].type(torch.bool))
    random_slice = nonzero_mask[np.random.randint(0, nonzero_mask.shape[0])]
    if is_color_channel:
        _, _, x, y, z = random_slice
        slices = [np.s_[0, 0, x, :, :], np.s_[0, 0, :, y, :], np.s_[0, 0, :, :, z]]
    else:
        _, x, y, z = random_slice
        slices = [np.s_[0, x, :, :], np.s_[0, :, y, :], np.s_[0, :, :, z]]

    for i, (key, label) in enumerate(zip(keys, labels)):
        plt.figure(figsize=(12, 4), constrained_layout=True)
        plt.suptitle(f'{label} {idx}')
        for j in range(3):
            plt.subplot(1, 3, j + 1)
            plt.title(f'Slice along {DIMS[j]}')
            plt.imshow(data_dict[key].detach()[slices[j]], cmap='Greys')


def rescale_volume(seismic):
    minval = np.percentile(seismic, 2)
    maxval = np.percentile(seismic, 98)
    if isinstance(seismic, MetaTensor):
        seismic = torch.clip(seismic, minval, maxval)
    else:
        seismic = np.clip(seismic, minval, maxval)
    seismic = ((seismic - minval) / (maxval - minval)) * 255
    return seismic

def training_data_generator(seismic, axis: Literal['i_line', 'x_line', None]=None, percentile: int=25):
    """Function to delete part of original seismic volume and extract target region

    Parameters:
        seismic: np.ndarray 3D matrix with original survey
        axis: one of 'i_line','x_line' or None. Axis along which part of survey will be deleted.
              If None (default), random will be chosen
        percentile: int, size of deleted part relative to axis. Any integer between 1 and 99 (default 20)

    Returns:
        seismic: np.ndarray, original survey 3D matrix with deleted region
        target: np.ndarray, 3D deleted region
        target_mask: np.ndarray, position of target 3D matrix in seismic 3D matrix.
                     This mask is used to reconstruct original survey -> seismic[target_mask]=target.reshape(-1)
    """

    # check parameters
    assert isinstance(seismic, np.ndarray) or isinstance(seismic, MetaTensor) and len(seismic.shape)==3, 'seismic must be 3D numpy.ndarray'
    assert axis in ['i_line', 'x_line', None], 'axis must be one of: i_line, x_line or None'
    assert type(percentile) is int and 0<percentile<100, 'percentile must be an integer between 0 and 100'

    # rescale volume
    seismic = rescale_volume(seismic)

    # if axis is None get random choice
    if axis is None:
        axis = np.random.choice(['i_line', 'x_line'], 1)[0]

    # crop subset
    if axis == 'i_line':
        sample_size = np.round(seismic.shape[0]*(percentile/100)).astype('int')
        sample_start = np.random.choice(range(seismic.shape[0]-sample_size), 1)[0]
        sample_end = sample_start+sample_size

        target_mask = np.zeros(seismic.shape).astype('bool')
        target_mask[sample_start:sample_end, :, :] = True

        target = seismic[sample_start:sample_end, :, :].copy()
        seismic[target_mask] = np.nan

    else:
        sample_size = np.round(seismic.shape[1]*(percentile/100)).astype('int')
        sample_start = np.random.choice(range(seismic.shape[1]-sample_size), 1)[0]
        sample_end = sample_start+sample_size

        target_mask = np.zeros(seismic.shape).astype('bool')
        target_mask[:, sample_start:sample_end, :] = True

        target = seismic[:, sample_start:sample_end, :].copy()
        seismic[target_mask] = np.nan

    return seismic, target, target_mask


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


class RescaleTransform(MapTransform):
    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = rescale_volume(data[key])
        return data


class CorruptedTransform(MapTransform):
    '''
    Transform that is applied on the original image and creates
    its corrupted version
    '''
    def __init__(self, percentile, **kwargs):
        super().__init__(**kwargs)
        self.percentile = percentile

    def __call__(self, data):
        if 'target' in data:
            data['image'], data['mask'] = training_data_generator_pt(data['target'], percentile=self.percentile)
        return data
