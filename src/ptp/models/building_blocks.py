import torch
from torch import nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        super(ConvBNReLU, self).__init__(
            nn.Conv3d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ResNetBlock3D(nn.Module):
    def __init__(self, c_in, act_fn, subsample=False, c_out=-1):
        """ResNetBlock.

        Args:
            c_in: Number of input features
            act_fn: Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample:
            c_out = c_in

        # Network representing F
        self.net = nn.Sequential(
            nn.Conv3d(
                c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False
            ),  # No bias needed as the Batch Norm handles it
            nn.BatchNorm3d(c_out),
            act_fn(),
            nn.Conv3d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(c_out),
        )

        # 1x1 convolution with stride 2 means we take the upper left value, and transform it to new output size
        self.downsample = nn.Conv3d(c_in, c_out, kernel_size=1, stride=2) if subsample else None
        self.act_fn = act_fn()

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        out = self.act_fn(out)
        return out


if __name__ == "__main__":



    resnet_block = ResNetBlock3D(1, nn.ReLU)
    resnet_input = torch.randn((1, 1, 10, 10, 10))
    print(resnet_input[0, 0, 0, 0, :])
    print(torch.mean(resnet_input))
    resnet_result = resnet_block(resnet_input)
    print(resnet_result.shape)
    print(resnet_result[0, 0, 0, 0, :])
    print(torch.mean(resnet_result))
