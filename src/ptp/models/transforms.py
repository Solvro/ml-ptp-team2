import numpy as np  # linear algebra
import torch
# Structural Similarity Loss
from monai.data import MetaTensor
from monai.transforms import MapTransform
from src.ptp.models.missing_volume_gen import training_data_generator_pt
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    SpatialCropd,
    SignalFillEmptyd,
    RandSpatialCropd,
    ScaleIntensityd
)

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


def get_transforms(percentile):
    return [LoadImaged(keys=['target']),
                 RescaleTransform(keys=['target']),
                 EnsureChannelFirstd(keys=['target']),
                 Orientationd(keys=["target"], axcodes="RAS"),
                 SpatialCropd(keys=['target'], roi_center=(150, 150, 750), roi_size=(256, 256, 256)),
                 CorruptedTransform(percentile=percentile, keys=['target']),
                 # Problem: missing areas are nans, which causes everything else to be nan
                 SignalFillEmptyd(keys=['image'], replacement=256),
                 # Scale to range [0,1]
                 ScaleIntensityd(keys=["image", "target"])]
