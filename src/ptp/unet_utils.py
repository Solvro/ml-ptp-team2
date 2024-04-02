unet_utils.py

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch

from monai.data import DataLoader, Dataset as MONAIDataset, MetaTensor, list_data_collate
from monai.losses import SismLoss
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import Compose, Orientationd, ScaleIntensityd
from torch.utils.data import Dataset


def training_data_generator2(seismic: np.ndarray, xy: Literal['x', 'y'], percentile: int = 5):
    """Function to delete part of original seismic volume ('i_line')

    Parameters:
        seismic: np.ndarray 3D matrix with original survey
        xy: str,
        percentile: int, size of deleted part relative to axis. Any integer between 1 and 99 (default 20)

    Returns:
        seismic: np.ndarray, original survey 3D matrix rescaled with deleted region if xy==x
    """

    # check parameters
    assert isinstance(seismic, np.ndarray) and len(seismic.shape) == 3, 'seismic must be 3D numpy.ndarray'
    assert type(percentile) is int and 0 < percentile < 100, 'percentile must be an integer between 0 and 100'

    # rescale volume
    minval = np.percentile(seismic, 2)
    maxval = np.percentile(seismic, 98)
    seismic = np.clip(seismic, minval, maxval)
    seismic = ((seismic - minval) / (maxval - minval)) * 255

    if xy == 'x':
        sample_size = np.round(seismic.shape[0] * (percentile / 100)).astype('int')
        sample_start = np.random.choice(range(seismic.shape[0] - sample_size), 1)[0]
        sample_end = sample_start + sample_size

        target_mask = np.zeros(seismic.shape).astype('bool')
        target_mask[sample_start:sample_end, :, :] = True

        target = seismic[sample_start:sample_end, :, :].copy()
        seismic[target_mask] = 255

    return seismic


class ToMetaTensor:
    def __call__(self, inputs, targets):
        return MetaTensor(x=torch.unsqueeze(torch.Tensor(inputs), 0)), MetaTensor(
            x=torch.unsqueeze(torch.Tensor(targets), 0)
        )


class PtpDataset(Dataset):
    def __init__(self, data_dir, transform=ToMetaTensor()):
        train_generated_data = [nib.load(path).get_fdata() for path in list(Path(data_dir).glob('*.nii.gz'))]
        self.x = list(map(lambda data: training_data_generator2(data, xy='x'), train_generated_data))
        self.y = list(map(lambda data: training_data_generator2(data, xy='y'), train_generated_data))
        self.transform = transform

    def __getitem__(self, index):
        inputs = self.x[index]
        outputs = self.y[index]
        if self.transform:
            inputs, outputs = self.transform(inputs, outputs)
        return inputs, outputs

    def __len__(self):
        return len(self.x)


class Net(pl.LightningModule):
    def __init__(self, dataset):
        super().__init__()

        self._model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
        self.dataset = dataset
        self.sism_loss = SismLoss()

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        return self._model(x)

    def prepare_data(self):
        images = self.dataset.x
        targets = self.dataset.y
        data_dicts = [{'image': image, 'target': target} for image, target in zip(images, targets)]
        train_dict, val_dict = data_dicts[:20], data_dicts[20:]

        train_transforms = Compose(
            [Orientationd(keys=['image', 'target'], axcodes='RAS'), ScaleIntensityd(keys=['image', 'target'])]
        )

        val_transforms = Compose(
            [Orientationd(keys=['image', 'target'], axcodes='RAS'), ScaleIntensityd(keys=['image', 'target'])]
        )

        self.train_data = MONAIDataset(data=train_dict, transform=train_transforms)

        self.val_data = MONAIDataset(data=val_dict, transform=val_transforms)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_data, batch_size=3, shuffle=True, collate_fn=list_data_collate)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_data, batch_size=1)
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), 1e-4)
        return optimizer

    def validation_step(self, batch):
        images, targets = batch['image'], batch['target']
        output = self.forward(images)
        loss = self.sism_loss(output, targets)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return {'val_loss': loss}

    def training_step(self, batch):
        images, targets = batch['image'], batch['target']
        output = self.forward(images)
        loss = self.sism_loss(output, targets)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return {'loss': loss}

    def visualize_results(self, seismic):
        missing_seismic, original_seismic = seismic
        original_volume = torch.Tensor(original_seismic)
        original_volume = self.forward(original_volume)

        x_slice = original_volume[:, :, original_volume.shape[2] // 2, :, :].squeeze().detach().numpy()
        y_slice = original_volume[:, :, :, original_volume.shape[3] // 2, :].squeeze().detach().numpy()
        z_slice = original_volume[:, :, :, :, original_volume.shape[4] // 2].squeeze().detach().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(x_slice, cmap='gray')
        axes[0].set_title('Slice x')
        axes[1].imshow(y_slice, cmap='gray')
        axes[1].set_title('Slice y')
        axes[2].imshow(z_slice, cmap='gray')
        axes[2].set_title('Slice z')
        plt.suptitle('Original Seismic')
        plt.show()

        missing_volume = torch.Tensor(missing_seismic)
        missing_volume = self.forward(missing_volume)

        x_slice = missing_volume[:, :, missing_volume.shape[2] // 2, :, :].squeeze().detach().numpy()
        y_slice = missing_volume[:, :, :, missing_volume.shape[3] // 2, :].squeeze().detach().numpy()
        z_slice = missing_volume[:, :, :, :, missing_volume.shape[4] // 2].squeeze().detach().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(x_slice, cmap='gray')
        axes[0].set_title('Slice x')
        axes[1].imshow(y_slice, cmap='gray')
        axes[1].set_title('Slice y')
        axes[2].imshow(z_slice, cmap='gray')
        axes[2].set_title('Slice z')
        plt.suptitle('Missing Seismic')
        plt.show()
