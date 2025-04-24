import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    Channel attention focuses on 'which' features are important,
    by assigning weights to each channel.
    """
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # A small MLP that transforms channel descriptors
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    Spatial attention focuses on 'where' features are important,
    by assigning weights to each position (H×W).
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply average pooling and max pooling along the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate along the channel axis
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """
    CBAM = Channel Attention + Spatial Attention
    """
    def __init__(self, channels, reduction=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.ChannelAtt = ChannelAttention(channels, reduction=reduction)
        self.SpatialAtt = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x):
        # Channel attention
        x_out = self.ChannelAtt(x) * x
        # Spatial attention
        x_out = self.SpatialAtt(x_out) * x_out
        return x_out


class ResidualBlock(nn.Module):
    """
    A basic residual block with two convolutional layers.
    If the number of input and output channels differ, a downsample (projection)
    layer is applied to the identity.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class UpBlock(nn.Module):
    """
    Decoder block: 
      1) Upsample (ConvTranspose2D),
      2) Concatenate skip connection,
      3) Convolution + BN + ReLU,
      4) CBAM for attention.
    """
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # After concatenation, the channels become (out_channels + skip_channels),
        # but we typically define the skip channels to match out_channels in the design.
        # If you use the same #filters in skip connections, 
        # then total channels = out_channels * 2 for the conv below.
        # Adjust as needed if your skip connections differ.
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        # CBAM module
        self.cbam = CBAM(out_channels)

    def forward(self, x, skip):
        # 1) Upsample
        x = self.up(x)

        # 2) If necessary, pad to match skip size (in case of odd/even dimension mismatches)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        # 3) Concatenate with skip
        x = torch.cat([skip, x], dim=1)

        # 4) Conv + BN + ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 5) CBAM
        x = self.cbam(x)

        return x


class ResUnet_CBAM(nn.Module):
    """
    ResUnet with CBAM in the decoder.
    Adjust the number of filters, layers, etc. to your needs.
    """
    def __init__(self, in_channels=3, out_channels=1, filters=[64, 128, 256, 512]):
        super(ResUnet_CBAM, self).__init__()

        # ---- 1) Initial Convolution ----
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )

        # ---- 2) Encoder ----
        # Block 1
        self.enc1 = ResidualBlock(filters[0], filters[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.enc2 = ResidualBlock(filters[0], filters[1],
                                  downsample=nn.Sequential(
                                      nn.Conv2d(filters[0], filters[1], kernel_size=1, bias=False),
                                      nn.BatchNorm2d(filters[1])
                                  ))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.enc3 = ResidualBlock(filters[1], filters[2],
                                  downsample=nn.Sequential(
                                      nn.Conv2d(filters[1], filters[2], kernel_size=1, bias=False),
                                      nn.BatchNorm2d(filters[2])
                                  ))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4 (bottom)
        self.enc4 = ResidualBlock(filters[2], filters[3],
                                  downsample=nn.Sequential(
                                      nn.Conv2d(filters[2], filters[3], kernel_size=1, bias=False),
                                      nn.BatchNorm2d(filters[3])
                                  ))

        # ---- 3) Decoder (with CBAM) ----
        self.up3 = UpBlock(filters[3], filters[2])  # Skip from enc3
        self.up2 = UpBlock(filters[2], filters[1])  # Skip from enc2
        self.up1 = UpBlock(filters[1], filters[0])  # Skip from enc1

        # ---- 4) Final 1x1 Conv for segmentation ----
        self.out_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # ---- Encoder ----
        x1 = self.initial(x)   # (B, 64, H, W)
        e1 = self.enc1(x1)     # (B, 64, H, W)
        p1 = self.pool1(e1)    # (B, 64, H/2, W/2)

        e2 = self.enc2(p1)     # (B, 128, H/2, W/2)
        p2 = self.pool2(e2)    # (B, 128, H/4, W/4)

        e3 = self.enc3(p2)     # (B, 256, H/4, W/4)
        p3 = self.pool3(e3)    # (B, 256, H/8, W/8)

        e4 = self.enc4(p3)     # (B, 512, H/8, W/8)

        # ---- Decoder ----
        d3 = self.up3(e4, e3)  # (B, 256, H/4, W/4)
        d2 = self.up2(d3, e2)  # (B, 128, H/2, W/2)
        d1 = self.up1(d2, e1)  # (B, 64,  H,   W)

        out = self.out_conv(d1)  # (B, out_channels, H, W)
        return out

