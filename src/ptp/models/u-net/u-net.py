import os

import torch
from monai.data import Dataset, DataLoader, list_data_collate
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    NormalizeIntensity,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    SpatialCropd,
    MapTransform,
    SignalFillEmptyd,
    RandSpatialCropd,
    ScaleIntensityd,
    ScaleIntensityRanged, MaskIntensity
)
from torch import nn
import pytorch_lightning as pl

from src.ptp.globals import TARGET_DATA_DIR

from src.ptp.models.transforms import RescaleTransform, CorruptedTransform


class Net(pl.LightningModule):

    def __init__(self, percentile=25, loss_func=nn.MSELoss()):
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

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        targets = sorted(os.listdir(TARGET_DATA_DIR))
        train_dict = [
            {'target': TARGET_DATA_DIR / target_name} for target_name in targets[:1]
        ]

        val_dict = [
            {'target': TARGET_DATA_DIR / target_name} for target_name in targets[:1]
        ]

        # TODO: Examine what these transformations do and whether some other transformations can be applied
        train_transforms = Compose(
            # for now each image would be normalized based on its own mean and std
            [LoadImaged(keys=['target']),
             RescaleTransform(keys=['target']),
             EnsureChannelFirstd(keys=['target']),
             Orientationd(keys=["target"], axcodes="RAS"),
             RandSpatialCropd(keys=['target'],
                              roi_size=(256, 256, 256), random_size=False),
             CorruptedTransform(percentile=self.percentile, keys=['target']),
             # Problem: missing areas are nans, which causes everything else to be nan
             SignalFillEmptyd(keys=['image'], replacement=256),
             ScaleIntensityd(keys=["image", "target"]),
             ]
        )

        val_transforms = Compose(
            # for now each image would be normalized based on its own mean and std
            [LoadImaged(keys=['target']),
             RescaleTransform(keys=['target']),
             EnsureChannelFirstd(keys=['target']),
             Orientationd(keys=["target"], axcodes="RAS"),
             SpatialCropd(keys=['target'], roi_center=(150, 150, 750),
                          roi_size=(256, 256, 256)),
             CorruptedTransform(percentile=self.percentile, keys=['target']),
             # Problem: missing areas are nans, which causes everything else to be nan
             SignalFillEmptyd(keys=['image'], replacement=256),
             # Scale to range [0,1]
             ScaleIntensityd(keys=["image", "target"]),
             ]
        )

        # Cached Dataset: na RAM wrzuca; to co jest wynikiem transformow wrzuca na RAM
        # ale to wtedy nie robi tego transform
        # Smart Cached Dataset: madre?
        self.train_data = Dataset(
            data=train_dict,
            transform=train_transforms
        )

        self.val_data = Dataset(
            data=val_dict,
            transform=val_transforms
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
