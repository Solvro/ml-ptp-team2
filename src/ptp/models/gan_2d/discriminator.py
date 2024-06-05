import pytorch_lightning as pl
import torch
from torch import nn

from src.ptp.models.utils import num_trainable_params


# TODO: spectral normalization?
class Discriminator(pl.LightningModule):

    def __init__(self, input_channels):
        super().__init__()
        # input: 1, 256, 256, 256
        self.layers = nn.ModuleList([
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=4, padding=1, bias=False),  # 16, 128, 128, 128
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 32, 64, 64, 64
            nn.BatchNorm2d(32, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 64, 32, 32, 32
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 128, 16, 16, 16
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),  # 128, 8, 8, 8
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(128, 1, kernel_size=4, bias=False),  #
        ])

    def forward(self, x):
        batch_size = x.shape[0]
        for l in self.layers:
            x = l(x)
        x = x.view(batch_size, -1)
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    discriminator = Discriminator(1)
    input_pt = torch.rand((10, 1, 256, 256))
    print(discriminator(input_pt).shape)
    print(num_trainable_params(discriminator))
