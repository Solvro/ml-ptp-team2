import os

import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from src.ptp.globals import TARGET_DATA_DIR
from src.ptp.models.gan.discriminator import Discriminator
from src.ptp.models.gan.generator import Generator
from monai.data import CacheDataset, Dataset, DataLoader, list_data_collate, NibabelReader
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
from monai.data import MetaTensor
from monai.apps.reconstruction.transforms.dictionary import ReferenceBasedNormalizeIntensityd

from src.ptp.models.transforms import RescaleTransform, CorruptedTransform, RandomNoiseTransform
from src.ptp.training.data_preparation import prepare_files_dirs


class GAN(pl.LightningModule):

    def __init__(self, percentile=20):
        super().__init__()
        self.G = Generator()
        self.D = Discriminator(1)

        self.percentile = percentile
        self.transforms = [LoadImaged(keys=['target']),
             NormalizeIntensity(keys=['target']),
             EnsureChannelFirstd(keys=['target']),
             Orientationd(keys=["target"], axcodes="RAS"),
             RandSpatialCropd(keys=['target'],
                              roi_size=(256, 256, 256), random_size=False),
             CorruptedTransform(percentile=self.percentile, keys=['target']),
             # Problem: missing areas are nans, which causes everything else to be nan
             SignalFillEmptyd(keys=['image'], replacement=0.0),
             ScaleIntensityd(keys=["image", "target"]),
             RandomNoiseTransform(keys=['image', 'mask'])
             ]


        # Activate manual optimization
        self.automatic_optimization = False

    def forward(self, x):
        return self.G(x)

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        X, targets, mask = batch["image"].as_tensor(), batch["target"].as_tensor(), batch['mask']
        batch_size = X.shape[0]

        # TODO: add some noise to the labels
        real_label = torch.ones((batch_size, 1), device=self.device)
        fake_label = torch.zeros((batch_size, 1), device=self.device)

        g_X = self.G(X)

        #################
        # Discriminator #
        #################
        d_x = self.D(targets)

        errD_real = F.binary_cross_entropy(d_x, real_label)

        # We need to detach because we don't want to optimize generator at the same time
        # Tensor.detach() creates new tensor detached from the computational graph
        d_z = self.D(g_X.detach())

        errD_fake = F.binary_cross_entropy(d_z, fake_label)

        # Compute the mean or not? - maybe compute the loss at the same time
        errD = (errD_real + errD_fake) / 2

        # We optimize only discriminator's parameters
        d_opt.zero_grad()
        self.manual_backward(errD)
        d_opt.step()

        #############
        # Generator #
        #############
        d_z = self.D(g_X)

        # discriminator should predict those as real
        errG_pred = F.binary_cross_entropy(d_z, real_label)
        # reconstruction loss
        errG_mse = F.mse_loss(g_X, targets)

        errG = (errG_pred + errG_mse) / 2

        g_opt.zero_grad()
        self.manual_backward(errG)
        g_opt.step()

        self.log_dict({'g_loss': errG, 'd_loss': errD, 'train_loss': (errG + errD)}, prog_bar=True, on_epoch=True)
        self.discriminator_real_loss.append(errD_real)
        self.discriminator_fake_loss.append(errD_fake)
        self.generator_real_loss.append(errG_pred)
        self.generator_mse_loss.append(errG_mse)


    def validation_step(self, batch, batch_idx):
        X, targets = batch["image"].as_tensor(), batch["target"].as_tensor()
        batch_size = X.shape[0]

        real_label = torch.ones((batch_size, 1), device=self.device)
        fake_label = torch.zeros((batch_size, 1), device=self.device)

        g_X = self.G(X)

        #################
        # Discriminator #
        #################
        d_x = self.D(targets)
        errD_real = F.binary_cross_entropy(d_x, real_label)

        # We need to detach because we don't want to optimize generator at the same time
        # Tensor.detach() creates new tensor detached from the computational graph
        d_z = self.D(g_X.detach())
        errD_fake = F.binary_cross_entropy(d_z, fake_label)

        # Compute the mean or not? - maybe compute the loss at the same time
        errD = (errD_real + errD_fake) / 2.0

        #############
        # Generator #
        #############
        d_z = self.D(g_X)
        # discriminator should predict those as real
        errG_pred = F.binary_cross_entropy(d_z, real_label)
        errG_mse = F.mse_loss(g_X, targets)

        errG = errG_pred + errG_mse

        self.log_dict({'g_loss': errG, 'd_loss': errD, 'val_loss': (errG + errD)}, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        # Discriminator and generator need to be trained separately so they have different optimizers
        d_opt = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        g_opt = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        return g_opt, d_opt

    def prepare_data(self):
        train_dict, val_dict = prepare_files_dirs()

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
             SignalFillEmptyd(keys=['image'], replacement=0.0),
             ScaleIntensityd(keys=["image", "target"]),
             RandomNoiseTransform(keys=['image', 'mask'])
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
             SignalFillEmptyd(keys=['image'], replacement=0.0),
             ScaleIntensityd(keys=["image", "target"]),
             RandomNoiseTransform(keys=['image', 'mask']),
             ]
        )

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
            batch_size=1,
            shuffle=True,
            num_workers=4,
            collate_fn=list_data_collate
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_data, batch_size=1, num_workers=4)
        return val_loader
