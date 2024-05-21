import os

import pytorch_lightning as pl
import torch
from monai.data import Dataset, DataLoader, list_data_collate
from monai.networks.layers import Norm
from monai.networks.nets import UNet
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
from torch import nn

from src.ptp.globals import TARGET_DATA_DIR
from src.ptp.models.transforms import RescaleTransform, CorruptedTransform, get_transforms
from src.ptp.training.data_preparation import prepare_files_dirs


class Net(pl.LightningModule):

    def __init__(self, percentile, target_data_dir, loss_func=nn.MSELoss()):
        super().__init__()

        # How these shapes form?; try with different input sizes
        self._model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH
        )

        self.loss_function = loss_func
        self.percentile = percentile
        self.target_data_dir = target_data_dir

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        train_dict, val_dict = prepare_files_dirs(self.target_data_dir)

        transforms = get_transforms(self.percentile)
        self.train_data = Dataset(
            data=train_dict,
            transform=Compose(transforms)
        )
        self.val_data = Dataset(
            data=val_dict,
            transform=Compose(transforms)
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_data,
            batch_size=2,
            shuffle=True,
            num_workers=2,
            collate_fn=list_data_collate
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_data, batch_size=1, num_workers=4)
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), 1e-4)
        return optimizer

    def validation_step(self, batch, batch_idx):
        images, targets = batch["image"], batch["target"]
        output = self.forward(images)
        loss = self.loss_function(output, targets)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return {"val_loss": loss}

    def training_step(self, batch, batch_idx):
        images, targets, mask = batch["image"], batch["target"], batch['mask']
        output = self.forward(images)
        loss = self.loss_function(output, targets)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return {"loss": loss}
