import os
from typing import Literal

import numpy as np  # linear algebra
import torch
# Structural Similarity Loss
from monai.data import MetaTensor
# Structural Similarity Loss
from monai.transforms import MapTransform

from src.ptp.models.missing_volume_gen import training_data_generator_pt


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



def rescale_volume(seismic):
    minval = np.percentile(seismic, 2)
    maxval = np.percentile(seismic, 98)
    if isinstance(seismic, MetaTensor):
        seismic = torch.clip(seismic, minval, maxval)
    else:
        seismic = np.clip(seismic, minval, maxval)
    seismic = ((seismic - minval) / (maxval - minval)) * 255
    return seismic


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


class RandomNoiseTransform(MapTransform):

    def __init__(self, missing_factor=0.7, original_factor=0.3, **kwargs):
        super().__init__(**kwargs)
        self.missing_factor = missing_factor
        self.original_factor = original_factor

    def __call__(self, data):
        data['image'] = data['image'] + (torch.rand(*data['mask'][0].shape) * data['mask'][0] * self.missing_factor) + (
                torch.rand(*data['mask'][0].shape) * ~data['mask'][0] * self.original_factor)
        return data



def prepare_files_dirs(target_data_dir):
    targets = sorted(os.listdir(target_data_dir))

    train_dict = [
        {'target': target_data_dir / target_name} for target_name in targets[:1]
    ]

    val_dict = [
        {'target': target_data_dir / target_name} for target_name in targets[:1]
    ]

    return train_dict, val_dict
