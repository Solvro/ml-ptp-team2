from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
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

from src.ptp.evaluation.visualization import visualize_volumes
from src.ptp.models.gan_3d.discriminator import Discriminator
from src.ptp.models.gan_3d.generator import Generator
from src.ptp.models.transforms import CorruptedTransform, RandomNoiseTransform, RescaleTransform
from src.ptp.training.data_preparation import prepare_files_dirs
from torch.utils.tensorboard import SummaryWriter


class GAN3D(pl.LightningModule):

    # recon_loss : reconstruction loss, either mse or tce
    def __init__(self, target_data_dir, percentile=0.05, n_critic=4, recon_loss=F.mse_loss,
                 weight_clip=0.01, batch_size=1):
        super().__init__()
        self.G = Generator()
        self.D = Discriminator(1)
        self.n_critic = n_critic
        self.target_data_dir = target_data_dir
        self.recon_loss = recon_loss
        self.weight_clip = weight_clip
        self.percentile = percentile
        self.batch_size = batch_size
        self.transforms = [LoadImaged(keys=['target']),
                           # NormalizeIntensityd(keys=['target']),
                           RescaleTransform(keys=['target']),
                           EnsureChannelFirstd(keys=['target']),
                           Orientationd(keys=["target"], axcodes="RAS"),
                           RandSpatialCropd(keys=['target'],
                                            roi_size=(256, 256, 256), random_size=False),
                           CorruptedTransform(percentile=self.percentile, keys=['target']),
                           SignalFillEmptyd(keys=['image'], replacement=127.5), # replace nan values to be the mean
                           ScaleIntensityd(keys=["image", "target"], minv=-1, maxv=1),  # the output image should be between -1 and 1
                           RandomNoiseTransform(keys=['image', 'mask'], lower=-1, upper=1) # add some random noise
                           ]
        # Activate manual optimization
        self.automatic_optimization = False
        self.compt_step = 0
        self.writer = SummaryWriter()
        self.validation_step_outputs = []

    def forward(self, x):
        return self.G(x)

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        X_corrupted, X_original, mask = batch["image"].as_tensor(), batch["target"].as_tensor(), batch['mask']

        g_X = self.G(X_corrupted)

        #################
        # Discriminator #
        #################
        d_x = self.D(X_original)

        errD_real = torch.mean(d_x)  # error of predicting real input

        d_z = self.D(g_X)

        errD_fake = torch.mean(d_z)  # error of predicting generator's output as fake

        errD = errD_fake - errD_real  # same as -(errD_real - errD_fake)

        # We optimize only discriminator's parameters
        d_opt.zero_grad()
        self.manual_backward(errD)
        d_opt.step()

        # Parameter Weight Clipping for K-Lipshitz constraint
        for p in self.D.parameters():
            p.data.clamp_(-self.weight_clip, self.weight_clip)

        #############
        # Generator #
        #############
        if self.compt_step % self.n_critic == 0:
            g_X = self.G(X_corrupted)
            d_z = self.D(g_X)

            # discriminator should predict those as real
            errG_pred = -torch.mean(d_z)
            # reconstruction loss
            errG_recon = self.recon_loss(g_X, X_original)

            errG = errG_pred + errG_recon

            g_opt.zero_grad()
            self.manual_backward(errG)
            g_opt.step()

            self.log_dict({'g_loss': errG}, prog_bar=True, on_epoch=True)

        self.log_dict({'d_loss': errD, 'n_step': self.compt_step}, prog_bar=True, on_epoch=True)
        self.compt_step += 1

    def validation_step(self, batch, batch_idx):
        X, targets = batch["image"].as_tensor(), batch["target"].as_tensor()

        g_X = self.G(X)

        #################
        # Discriminator #
        #################
        d_x = self.D(targets)
        errD_real = torch.mean(d_x)

        # We need to detach because we don't want to optimize generator at the same time
        # Tensor.detach() creates new tensor detached from the computational graph
        d_z = self.D(g_X.detach())
        errD_fake = torch.mean(d_z)

        errD = errD_fake - errD_real

        #############
        # Generator #
        #############
        d_z = self.D(g_X)
        # discriminator should predict those as real
        errG_pred = -torch.mean(d_z)
        # reconstruction loss
        errG_mse = self.recon_loss(g_X, targets)

        errG = (errG_pred + errG_mse) / 2

        self.log_dict({'val_g_loss': errG, 'val_d_loss': errD, 'val_loss': (errG + errD)}, prog_bar=True, on_epoch=True)
        self.validation_step_outputs.append(g_X)

    def on_validation_epoch_end(self) -> None:
        val_outputs = torch.stack(self.validation_step_outputs)
        print(val_outputs.shape)
        grid_x = torchvision.utils.make_grid(val_outputs[:, 0, :, 100, :, :])
        self.writer.add_image('images - x', grid_x, 0)
        grid_y = torchvision.utils.make_grid(val_outputs[:, 0, :, :, 100, :])
        self.writer.add_image('images - y', grid_y, 0)
        grid_z = torchvision.utils.make_grid(val_outputs[:, 0, :, :, :, 100])
        self.writer.add_image('images - z', grid_z, 0)
        self.writer.close()
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # Discriminator and generator need to be trained separately so they have different optimizers
        d_opt = torch.optim.RMSprop(self.D.parameters(), lr=0.0004)
        g_opt = torch.optim.RMSprop(self.G.parameters(), lr=0.0001)
        return g_opt, d_opt

    def prepare_data(self):
        one_sample_mode = (self.batch_size == 1)
        train_dict, val_dict = prepare_files_dirs(self.target_data_dir, one_sample_mode=one_sample_mode)

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
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=list_data_collate
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_data, batch_size=self.batch_size, num_workers=4)
        return val_loader


if __name__ == '__main__':
    data_dir = Path("C://Users//julia//PycharmProjects//SOLVRO-PTP2//data//generated_part1_nii_gz")
    gan3d = GAN3D(data_dir, 20, 4)

    gan3d.prepare_data()
    train_loader = gan3d.train_dataloader()

    keys = ['target', 'image', 'prediction']
    labels = ['Original', 'Corrupted', 'Prediction']

    checkpoint_dir = Path('C://Users//julia//PycharmProjects//SOLVRO-PTP2//data//models//gan-test')

    trainer = pl.Trainer(
        max_epochs=1,
        default_root_dir=checkpoint_dir,
        fast_dev_run=True,
    )

    trainer.fit(gan3d)

    for i, batch in enumerate(train_loader):
        print(batch['image'].shape)
        prediction = gan3d(batch['image'])
        batch['prediction'] = prediction
        visualize_volumes(batch, i, keys, labels, True)
        plt.show()


