import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the BasicBlock used in ResNet-34
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

# Define a layer of multiple BasicBlocks
def make_layer(in_channels, out_channels, blocks, stride=1):
    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    layers = [BasicBlock(in_channels, out_channels, stride, downsample)]
    for _ in range(1, blocks):
        layers.append(BasicBlock(out_channels, out_channels))
    return nn.Sequential(*layers)

# Encoder with ResNet-34 style
class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.layer1 = make_layer(64, 64, blocks=3)
        self.layer2 = make_layer(64, 128, blocks=4, stride=2)
        self.layer3 = make_layer(128, 256, blocks=6, stride=2)
        self.layer4 = make_layer(256, 512, blocks=3, stride=2)

    def forward(self, x):
        x0 = self.initial(x)
        x1 = self.layer1(x0)  # 1/4
        x2 = self.layer2(x1)  # 1/8
        x3 = self.layer3(x2)  # 1/16
        x4 = self.layer4(x3)  # 1/32
        return x1, x2, x3, x4

# Decoder block (Upsample + Conv)
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def concat_skipconnec(skipped_connec, upsampled):

        # To concat skip connec and upsample
        diffY = skipped_connec.size()[2] - upsampled.size()[2]
        diffX = skipped_connec.size()[3] - upsampled.size()[3]
        upsampled_padded = F.pad(upsampled, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        return upsampled_padded
    
    def forward(self, x, skip):
        x_upsampled = self.up(x)
        x_upsampled_padded = self.concat_skipconnec(skip, x_upsampled)
        x_padded = torch.cat([x_upsampled_padded , skip], dim=1)
        return self.conv(x_padded)

# Full U-Net with ResNet-34 Encoder
class ResNetUNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.encoder = ResNetEncoder()

        # Decoder
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        # self.decoder1 = DecoderBlock(64, 64, 32)
        self.decoder1 = DecoderBlock(64, 3, 32)

        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        d4 = self.decoder4(x4, x3)
        d3 = self.decoder3(d4, x2)
        d2 = self.decoder2(d3, x1)
        d1 = self.decoder1(d2, x)

        return self.final_conv(d1)
