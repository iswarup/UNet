import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_channels, out_channels, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1,64)
        self.down_conv_1 = double_conv(64, 128)
        self.down_conv_1 = double_conv(128,256)
        self.down_conv_1 = double_conv(256, 512)
        self.down_conv_1 = double_conv(512, 1024)
