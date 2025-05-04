# model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings as in Vaswani et al.
    Args:
        timesteps: Tensor of shape (batch,)
        embedding_dim: dimension of the output embedding
    Returns:
        Tensor of shape (batch, embedding_dim)
    """
    half_dim = embedding_dim // 2
    exponent = -math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1))
    return emb


class ResidualBlock(nn.Module):
    """Residual block with time & cond embeddings."""
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.cond_mlp = nn.Linear(cond_emb_dim, out_channels)
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb, c_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        # add time and condition embeddings
        h = h + self.time_mlp(t_emb)[:, :, None, None] + self.cond_mlp(c_emb)[:, :, None, None]
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).reshape(B, C, -1)
        k = self.k(h).reshape(B, C, -1)
        v = self.v(h).reshape(B, C, -1)
        attn = torch.bmm(q.permute(0, 2, 1), k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class Down(nn.Module):
    """Downsampling block: Residual + Attention + Pool."""
    def __init__(self, in_ch, out_ch, time_emb_dim, cond_emb_dim):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch, time_emb_dim, cond_emb_dim)
        self.attn = AttentionBlock(out_ch)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x, t_emb, c_emb):
        x = self.res(x, t_emb, c_emb)
        x = self.attn(x)
        return self.pool(x)


class Up(nn.Module):
    """Upsampling block: TransposeConv + Residual + Attention."""
    def __init__(self, in_ch, skip_ch, out_ch, time_emb_dim, cond_emb_dim):
        super().__init__()
        # 1) upsample from in_ch → out_ch
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # 2) after concatenation, channels = out_ch + skip_ch
        self.res = ResidualBlock(
            in_channels=out_ch + skip_ch,
            out_channels=out_ch,
            time_emb_dim=time_emb_dim,
            cond_emb_dim=cond_emb_dim
        )
        # 3) attention on out_ch
        self.attn = AttentionBlock(out_ch)

    def forward(self, x, skip, t_emb, c_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb, c_emb)
        x = self.attn(x)
        return x


class AttentionUNet(nn.Module):
    """Conditional UNet with attention for DDPM training and inference."""
    def __init__(
        self,
        in_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_emb_dim=256,
        cond_emb_dim=256,
        out_channels=3
    ):
        super().__init__()
        # Embedding MLPs
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, cond_emb_dim)
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Build downsampling path
        channels = [base_channels * m for m in channel_mults]
        self.downs = nn.ModuleList()
        prev_ch = base_channels
        for ch in channels:
            self.downs.append(Down(prev_ch, ch, time_emb_dim, cond_emb_dim))
            prev_ch = ch

        # Middle blocks
        self.mid1 = ResidualBlock(prev_ch, prev_ch, time_emb_dim, cond_emb_dim)
        self.mid_attn = AttentionBlock(prev_ch)
        self.mid2 = ResidualBlock(prev_ch, prev_ch, time_emb_dim, cond_emb_dim)

        # Build upsampling path
        # skip_channels: [base_channels] + channels[:-1]
        skip_channels = [base_channels] + channels[:-1]
        rev_channels = channels[::-1]
        in_channels_up = [prev_ch] + rev_channels[:-1]

        self.ups = nn.ModuleList()
        for in_ch, skip_ch, out_ch in zip(in_channels_up, skip_channels[::-1], rev_channels):
            self.ups.append(
                Up(in_ch, skip_ch, out_ch, time_emb_dim, cond_emb_dim)
            )
            prev_ch = out_ch

        # Final normalization and output
        self.final_norm = nn.GroupNorm(8, base_channels)
        self.final_conv = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # 1) Embed time & condition
        t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)
        c_emb = self.cond_mlp(cond)

        # 2) Down path
        x = self.init_conv(x)
        skips = []
        for down in self.downs:
            skips.append(x)
            x = down(x, t_emb, c_emb)

        # 3) Middle
        x = self.mid1(x, t_emb, c_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb, c_emb)

        # 4) Up path
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip, t_emb, c_emb)

        # 5) Final
        x = self.final_norm(x)
        x = F.silu(x)
        return self.final_conv(x)
