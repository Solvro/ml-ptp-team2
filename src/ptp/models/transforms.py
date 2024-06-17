import numpy as np  # linear algebra
import torch
# Structural Similarity Loss
from monai.data import MetaTensor
from monai.transforms import MapTransform

from src.ptp.models.missing_volume_gen import training_data_generator_pt


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
    '''
    Transform that scales the original data based on 2nd and 98th quantile
    to a range between 0 and 255
    '''
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

    def __init__(self, missing_factor=0.7, original_factor=0.3, lower=0, upper=1, **kwargs):
        super().__init__(**kwargs)
        self.missing_factor = missing_factor
        self.original_factor = original_factor
        self.lower = lower
        self.upper = upper

    def __call__(self, data):
        data['image'] = torch.clip(data['image'] + (torch.rand(*data['mask'][0].shape) * data['mask'][0] * self.missing_factor) + (
                torch.rand(*data['mask'][0].shape) * ~data['mask'][0] * self.original_factor), min=self.lower, max=self.upper)
        return data


class SliceTransform(MapTransform):

    def __init__(self, low=0, high=256, **kwargs):
        super().__init__(**kwargs)
        self.low = low
        self.high = high

    def __call__(self, data):
        z = np.random.randint(self.low, self.high)
        data['image'] = data['image'][:, :, :, z]
        data['mask'] = data['mask'][:, :, :, z]
        data['target'] = data['target'][:, :, :, z]
        return data
