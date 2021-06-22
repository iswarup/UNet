import torch
import torch.nn as nn
import warnings
warnings.simplefilter("ignore", UserWarning)

def double_conv(in_channels, out_channels):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_channels, out_channels, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv

def crop_image(input_tensor,target_tensor):
    input_size = input_tensor.size()[2]
    target_size = target_tensor.size()[2]
    delta = input_size - target_size
    delta = delta // 2  
    return input_tensor[:, :, delta:input_size-delta, delta:input_size-delta]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1,64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128,256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv_1 = double_conv(1024,512)

        self.up_trans_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv_2 = double_conv(512,256)

        self.up_trans_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv_3 = double_conv(256,128)

        self.up_trans_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv_4 = double_conv(128,64)

        self.out = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

    def forward(self, image):
        # encoder
        x1 = self.down_conv_1(image)
        x1_pooled = self.max_pool_2x2(x1)

        x2 = self.down_conv_2(x1_pooled)
        x2_pooled = self.max_pool_2x2(x2)
        
        x3 = self.down_conv_3(x2_pooled)
        x3_pooled = self.max_pool_2x2(x3)

        x4 = self.down_conv_4(x3_pooled)
        x4_pooled = self.max_pool_2x2(x4)

        x5 = self.down_conv_5(x4_pooled)

        # print("Encoder output x5 size: ",x5.size())  # 1024

        # decoder
        x = self.up_trans_1(x5)                 # 512
        y = crop_image(x4,x)
        x = self.up_conv_1(torch.cat([x,y],1))

        x = self.up_trans_2(x)
        y = crop_image(x3,x)
        x = self.up_conv_2(torch.cat([x,y],1))

        x = self.up_trans_3(x)
        y = crop_image(x2,x)
        x = self.up_conv_3(torch.cat([x,y],1))

        x = self.up_trans_4(x)
        y = crop_image(x1,x)
        x = self.up_conv_4(torch.cat([x,y],1))

        x = self.out(x)

        return x

if __name__ == "__main__":
    image = torch.rand(1,1,572,572)
    model = UNet()
    print(model(image).size())