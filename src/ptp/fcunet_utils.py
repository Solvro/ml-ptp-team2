from pathlib import Path
from typing import Literal
import torch.nn as nn
from torchmetrics.functional.image import structural_similarity_index_measure
import torch.nn.functional as F

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch

from monai.data import DataLoader, Dataset as MONAIDataset, MetaTensor, list_data_collate
from monai.networks.layers import Norm
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
        target_mask[sample_start:sample_end, :] = True

        target = seismic[sample_start:sample_end, :].copy()
        seismic[target_mask] = 255

    return seismic


class Dataset_2d(Dataset):
    def __init__(self, data_dir):
        train_generated_data = [nib.load(path).get_fdata() for path in list(Path(data_dir).glob('*.nii.gz'))]
        self.x = [torch.tensor(np.expand_dims(training_data_generator2(data, xy='x'), axis=0)).float() for data in train_generated_data]
        self.y = [torch.tensor(np.expand_dims(training_data_generator2(data, xy='y'), axis=0)).float() for data in train_generated_data]

    def __getitem__(self, index):
        inputs = self.x[index]
        outputs = self.y[index]
        return inputs, outputs

    def __len__(self):
        return len(self.x)


def ssim_loss(preds, target):
    return 1 - structural_similarity_index_measure(preds, target, data_range=1.0)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers):
        super(ConvBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_conv_layers - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(out_channels))
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)

class FCUnet(pl.LightningModule):
    def __init__(self, dataset):
        super(FCUnet, self).__init__()
        self.dataset = dataset
        self.sism_loss = ssim_loss  # Pass ssim_loss function without calling it

        self.l1_3 = ConvBlock(in_channels=1, out_channels=32, num_conv_layers=3)
        self.l1_7 = ConvBlock(in_channels=32, out_channels=32, num_conv_layers=4)
        self.l2_4 = ConvBlock(in_channels=32, out_channels=64, num_conv_layers=3)

        self.l1_8 = ConvBlock(in_channels=32, out_channels=32, num_conv_layers=1)
        self.l2_5 = ConvBlock(in_channels=64, out_channels=64, num_conv_layers=1)

        self.l1_11 = ConvBlock(in_channels=96, out_channels=32, num_conv_layers=3)

        self.l2_8 = ConvBlock(in_channels=96, out_channels=64, num_conv_layers=3)

        self.l3_4 = ConvBlock(in_channels=96, out_channels=128, num_conv_layers=3)

        self.l1_12 = ConvBlock(in_channels=32, out_channels=32, num_conv_layers=1)

        self.l2_9 = ConvBlock(in_channels=64, out_channels=64, num_conv_layers=1)

        self.l1_15 = ConvBlock(in_channels=224, out_channels=32, num_conv_layers=3)

        self.l2_12 = ConvBlock(in_channels=224, out_channels=64, num_conv_layers=3)

        self.l1_16 = ConvBlock(in_channels=32, out_channels=32, num_conv_layers=1)

        self.l1_19 = ConvBlock(in_channels=96, out_channels=32, num_conv_layers=3)

        self.out = ConvBlock(in_channels=32, out_channels=1, num_conv_layers=1)

    def forward(self, x):
        l1_3_out = self.l1_3(x)
        l1_7_out = self.l1_7(l1_3_out)

        l2_1_out = F.max_pool2d(l1_3_out, kernel_size=2)
        l2_4_out = self.l2_4(l2_1_out)

        l1_8_out = torch.cat((self.l1_8(l1_7_out), F.interpolate(l2_4_out, scale_factor=2, mode='bilinear', align_corners=True)), dim=1)

        l2_5_out = torch.cat((F.max_pool2d(l1_7_out, kernel_size=2), self.l2_5(l2_4_out)), dim=1)

        l3_1_out = torch.cat((F.max_pool2d(F.max_pool2d(l1_7_out, kernel_size=2), kernel_size=2), F.max_pool2d(l2_4_out, kernel_size=2)), dim=1)

        l1_11_out = self.l1_11(l1_8_out)

        l2_8_out = self.l2_8(l2_5_out)

        l3_4_out = self.l3_4(l3_1_out)

        l1_12_out = torch.cat((self.l1_12(l1_11_out), F.interpolate(l2_8_out, scale_factor=2, mode='bilinear', align_corners=True), F.interpolate(F.interpolate(l3_4_out, scale_factor=2, mode='bilinear', align_corners=True), scale_factor=2, mode='bilinear', align_corners=True)), dim=1)

        l2_9_out = torch.cat((F.max_pool2d(l1_11_out, kernel_size=2), self.l2_9(l2_8_out), F.interpolate(l3_4_out, scale_factor=2, mode='bilinear', align_corners=True)), dim=1)

        l1_15_out = self.l1_15(l1_12_out)

        l2_12_out = self.l2_12(l2_9_out)

        l1_16_out = torch.cat((self.l1_16(l1_15_out), F.interpolate(l2_12_out, scale_factor=2, mode='bilinear', align_corners=True)), dim=1)

        l1_19_out = self.l1_19(l1_16_out)

        out = torch.sigmoid(self.out(l1_19_out))

        return out

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)  # Change _model to parameters
        return optimizer

    def validation_step(self, batch, batch_idx):
        images, targets = batch['image'], batch['target']
        output = self.forward(images)
        loss = self.sism_loss(output, targets)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return {'val_loss': loss}

    def training_step(self, batch, batch_idx):
        images, targets = batch['image'], batch['target']
        output = self.forward(images)
        loss = self.sism_loss(output, targets)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return {'loss': loss}

    def visualize_results(self, seismic):
        missing_seismic, original_seismic = seismic
        patched_volume = self.forward(missing_seismic)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(original_seismic[:, :], cmap='gray')
        axes[0].set_title("Original Seismic")
        axes[0].axis('off')

        axes[1].imshow(patched_volume[:, :], cmap='gray')
        axes[1].set_title("Patched Seismic")
        axes[1].axis('off')

        plt.show()
         





