import pytorch_lightning as pl
import torch
from torch import nn

from ptp.models.gan_2d.building_blocks import InvertedResidual2d, ResNetBlock2d
from ptp.models.utils import num_trainable_params


class Generator(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.down1 = nn.Sequential(
            # input: 1, 256, 256, 256
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 64, 128, 128, 128
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(0.2)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 64, 64, 64, 64
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(0.2)
        )

        self.inv_res_h_1 = InvertedResidual2d(64, oup=64, stride=1, expand_ratio=0.5)  # 64, 64, 64, 64

        # part to create lower resolution features
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 32, 32, 32, 32
            nn.BatchNorm2d(32, momentum=0.9),
            nn.LeakyReLU(0.2)
        )

        self.resnet_l_1 = ResNetBlock2d(32, act_fn=nn.ReLU)  # 32, 32, 32, 32

        self.resnet_h_1 = ResNetBlock2d(64, act_fn=nn.ReLU)  # 64, 64, 64, 64

        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(0.2)
        )  # 64, 64, 64, 64

        # concatenate to have 128, 64, 64, 64
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.BatchNorm2d(1, momentum=0.9),
            nn.LeakyReLU(0.2)
        )  # 1, ??

        self.norm = nn.Tanh()

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x_h = x.clone()

        # Higher dimensional features
        x_h = self.inv_res_h_1(x_h)
        x_h = self.resnet_h_1(x_h)

        # Lower dimensional features
        x_l = self.down3(x)
        x_l = self.resnet_l_1(x_l)
        x_l = self.upsampling(x_l)

        x = torch.cat((x_l, x_h), axis=1)
        x = self.up2(x)
        # Generator's output must be in the appropriate range!
        x = self.norm(x)
        return x


if __name__ == '__main__':
    gen = Generator()
    t1 = torch.randn((1, 1, 256, 256))
    output = gen(t1)
    print(output[0, 0, :5, :])
    print(output.shape)
    print(num_trainable_params(gen))
