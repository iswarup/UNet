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

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1,64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128,256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

    def forward(self, image):
        # encoder
        x1 = self.down_conv_1(image)
        x1 = self.max_pool_2x2(x1)

        x2 = self.down_conv_2(x1)
        x2 = self.max_pool_2x2(x2)
        
        x3 = self.down_conv_3(x2)
        x3 = self.max_pool_2x2(x3)

        x4 = self.down_conv_4(x3)
        x4 = self.max_pool_2x2(x4)

        x5 = self.down_conv_5(x4)

        print(x5.size())


if __name__ == "__main__":
    image = torch.rand(1,1,572,572)
    model = UNet()
    model(image)