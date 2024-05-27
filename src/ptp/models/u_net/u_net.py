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
    ScaleIntensityd
)
from torch import nn

from src.ptp.models.transforms import RescaleTransform, CorruptedTransform
from src.ptp.training.data_preparation import prepare_files_dirs


class Net(pl.LightningModule):

    def __init__(self, percentile,
                 target_data_dir, loss_func=nn.MSELoss(),
                one_sample_mode=False):
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
        self.transforms = [LoadImaged(keys=['target']),
                 RescaleTransform(keys=['target']),
                 EnsureChannelFirstd(keys=['target']),
                 Orientationd(keys=["target"], axcodes="RAS"),
                 SpatialCropd(keys=['target'], roi_center=(150, 150, 750), roi_size=(256, 256, 256)),
                 CorruptedTransform(percentile=percentile, keys=['target']),
                 # Problem: missing areas are nans, which causes everything else to be nan
                 SignalFillEmptyd(keys=['image'], replacement=256),
                 # Scale to range [0,1]
                 ScaleIntensityd(keys=["image", "target"])
        ]
        self.target_data_dir = target_data_dir
        self.one_sample_mode = one_sample_mode

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        train_dict, val_dict = prepare_files_dirs(self.target_data_dir, one_sample_mode=self.one_sample_mode)
        # TODO: Examine what these transformations do and whether some other transformations can be applied
        # Cached Dataset: na RAM wrzuca; to co jest wynikiem transformow wrzuca na RAM
        # ale to wtedy nie robi tego transform
        # Smart Cached Dataset: madre?
        self.train_data = Dataset(
            data=train_dict,
            transform=Compose(self.transforms)
        )
        self.val_data = Dataset(
            data=val_dict,
            transform=Compose(self.transforms)
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_data,
            batch_size=1 if self.one_sample_mode else 5,
            shuffle=True,
            num_workers=4,
            collate_fn=list_data_collate
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_data, batch_size=1 if self.one_sample_mode else 5, num_workers=2)
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
