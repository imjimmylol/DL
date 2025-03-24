# Implement your UNet model here
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.dcc = nn.Sequential(
            DoubleConv(in_channels=in_channels, out_channels=out_channels),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.dcc(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
    
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=2, stride=2) # down * 2 -> out // 2
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    @staticmethod
    def concat_skipconnec(skipped_connec, upsampled):

        # To concat skip connec and upsample
        diffY = skipped_connec.size()[2] - upsampled.size()[2]
        diffX = skipped_connec.size()[3] - upsampled.size()[3]
        upsampled_padded = F.pad(upsampled, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        return upsampled_padded

    def forward(self, x, skipped_connec):

        x_upsampled = self.up(x)
        x_upsampled_padded = self.concat_skipconnec(skipped_connec, x_upsampled)
        # print(skipped_connec.shape)
        # print(x_upsampled.shape)
        # print(x_upsampled_padded.shape)
        x = torch.cat([skipped_connec, x_upsampled_padded], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, n_classes=1):
        super().__init__()
        self.outputcov = nn.Conv2d(in_channels=in_channels, out_channels=n_classes,
                                   kernel_size=1, padding=0)
    def forward(self, x):
        return self.outputcov(x)

class Unet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024))
        self.up1 = (Up(1024, 512))
        self.up2 = (Up(512, 256))
        self.up3 = (Up(256, 128))
        self.up4 = (Up(128, 64))
        self.outc = (OutConv(64, n_classes))
        

    def forward(self, x):
        xd1 = self.inc(x)
        xd2 = self.down1(xd1)
        xd3 = self.down2(xd2)
        xd4 = self.down3(xd3)
        x_bottom = self.down4(xd4)

        xu1 = self.up1(x_bottom, xd4)
        xu2 = self.up2(xu1, xd3)
        xu3 = self.up3(xu2, xd2)
        xu4 = self.up4(xu3, xd1)

        out = self.outc(xu4)
        return out


