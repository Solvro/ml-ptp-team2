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
