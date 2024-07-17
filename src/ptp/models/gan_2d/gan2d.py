from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from monai.data import Dataset, DataLoader, list_data_collate
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    SignalFillEmptyd,
    RandSpatialCropd,
    ScaleIntensityd, NormalizeIntensityd
)

from ptp.evaluation.visualization import visualize_slices
from ptp.models.gan_2d.discriminator import Discriminator
from ptp.models.gan_2d.generator import Generator
from ptp.models.transforms import CorruptedTransform, RandomNoiseTransform, SliceTransform
from ptp.training.data_preparation import prepare_files_dirs


class GAN2D(pl.LightningModule):

    def __init__(self, percentile, target_data_dir, n_critic):
        super().__init__()
        self.G = Generator()
        self.D = Discriminator(1)
        self.n_critic = n_critic
        self.target_data_dir = target_data_dir

        self.percentile = percentile
        # Adjust to 2d transformations
        self.transforms = [LoadImaged(keys=['target']),
                           NormalizeIntensityd(keys=['target']),
                           EnsureChannelFirstd(keys=['target']),
                           Orientationd(keys=["target"], axcodes="RAS"),
                           RandSpatialCropd(keys=['target'],
                                            roi_size=(256, 256, 256), random_size=False),
                           CorruptedTransform(percentile=self.percentile, keys=['target']),
                           # Problem: missing areas are nans, which causes everything else to be nan
                           SignalFillEmptyd(keys=['image'], replacement=0.0),
                           ScaleIntensityd(keys=["image", "target"]),
                           RandomNoiseTransform(keys=['image', 'mask']),
                           SliceTransform(keys=['image', 'mask', 'target'])
                           ]

        # Activate manual optimization
        self.automatic_optimization = False
        self.compt_step = 0

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
        if self.compt_step % self.n_critic == 0:
            g_X = self.G(X)
            d_z = self.D(g_X)

            # discriminator should predict those as real
            errG_pred = F.binary_cross_entropy(d_z, real_label)
            # reconstruction loss
            # errG_mse = F.mse_loss(g_X, targets)

            # errG = (errG_pred + errG_mse) / 2
            errG = errG_pred

            g_opt.zero_grad()
            self.manual_backward(errG)
            g_opt.step()

            self.log_dict({'g_loss': errG}, prog_bar=True, on_epoch=True)

        self.log_dict({'d_loss': errD, 'n_step': self.compt_step}, prog_bar=True, on_epoch=True)
        self.compt_step += 1

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
        # errG_mse = F.mse_loss(g_X, targets)

        errG = errG_pred  # + errG_mse

        self.log_dict({'val_g_loss': errG, 'val_d_loss': errD, 'val_loss': (errG + errD)},
                      prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        # Discriminator and generator need to be trained separately so they have different optimizers
        d_opt = torch.optim.RMSprop(self.D.parameters(), lr=0.0002)
        g_opt = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        return g_opt, d_opt

    def prepare_data(self):
        train_dict, val_dict = prepare_files_dirs(self.target_data_dir, one_sample_mode=True)

        train_transforms = Compose(
            self.transforms
        )

        val_transforms = Compose(
            self.transforms
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
            batch_size=2,
            shuffle=True,
            num_workers=4,
            collate_fn=list_data_collate
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_data, batch_size=1, num_workers=4)
        return val_loader


if __name__ == '__main__':
    data_dir = Path("C://Users//julia//PycharmProjects//SOLVRO-PTP2//data//generated_part1_nii_gz")
    gan2d = GAN2D(20, data_dir, 4)

    gan2d.prepare_data()
    train_loader = gan2d.train_dataloader()

    keys = ['target', 'image', 'prediction']
    labels = ['Original', 'Corrupted', 'Prediction']

    for i, batch in enumerate(train_loader):
        print(batch['image'].shape)
        prediction = gan2d(batch['image'])
        batch['prediction'] = prediction
        visualize_slices(batch, i, keys, labels, True)
