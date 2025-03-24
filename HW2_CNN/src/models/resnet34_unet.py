# Implement your UNet model here
import torch
import torch.nn as nn
import torch.nn.functional as F


class encoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip Connection 的 1x1 映射
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)  # 映射 skip connection
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual  # 尺寸對齊後相加
        out = nn.ReLU()(out)
        return out


class up(nn.Module):

    def __init__(self, in_channels, out_channels=32):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels) # Error
        self.bn = nn.BatchNorm2d()
        s


    return None

# class Decoder(nn.Module):

#     def __init__(self, in_channels, out_channels):
#         super(Decoder, self).__init__()

#         self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=2, stride=2)
#         self.conv = nn.Conv2d(in_channels=out_channels//2, out_channels=)