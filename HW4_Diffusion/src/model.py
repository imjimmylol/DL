import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def sinusoidal_time_emb(timesteps, embedding_dim):
    """
    Create sinusoidal embeddings for timesteps.
    """
    device = timesteps.device
    half_dim = embedding_dim // 2
    emb = torch.exp(
        torch.arange(half_dim, device=device, dtype=torch.float32)
        * -(math.log(10000) / (half_dim - 1))
    )
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class TimeEmbedding(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t):
        return self.mlp(t)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch * 2)
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # time conditioning
        t_out = self.time_mlp(t_emb)
        scale, shift = t_out.chunk(2, dim=1)
        scale = scale[..., None, None]
        shift = shift[..., None, None]
        h = h * (1 + scale) + shift

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x):  # x: (B, C, H, W)
        b, c, h, w = x.shape
        x_in = x
        x_norm = self.norm(x)
        x_flat = x_norm.reshape(b, c, -1)  # (B, C, N)

        qkv = self.qkv(x_flat).chunk(3, dim=1)
        q, k, v = [t.reshape(b, self.num_heads, c // self.num_heads, -1) for t in qkv]

        scale = (c // self.num_heads) ** -0.5
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        out_flat = out.reshape(b, c, -1)
        out_proj = self.proj(out_flat).reshape(b, c, h, w)
        return out_proj + x_in


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_channels=128,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=128,
        num_heads=4
    ):
        super().__init__()
        self.num_res_blocks = num_res_blocks

        # time embedding
        self.time_emb = TimeEmbedding(time_emb_dim, time_emb_dim)

        # initial conv
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # down path
        in_ch = base_channels
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for mult in channel_mults:
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(in_ch, out_ch, time_emb_dim))
                in_ch = out_ch
            self.downsamples.append(Downsample(in_ch))

        # bottleneck
        self.mid_block1 = ResidualBlock(in_ch, in_ch, time_emb_dim)
        self.mid_attn = AttentionBlock(in_ch, num_heads)
        self.mid_block2 = ResidualBlock(in_ch, in_ch, time_emb_dim)

        # up path
        # We track in_ch carrying the current feature dims
        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            # upsample: reduce channels from in_ch -> out_ch
            self.upsamples.append(Upsample(in_ch, out_ch))
            # blocks: each takes concatenated skip (out_ch) and upsampled (out_ch) => 2*out_ch
            for _ in range(num_res_blocks):
                self.up_blocks.append(ResidualBlock(out_ch * 2, out_ch, time_emb_dim))
            in_ch = out_ch

        # final conv
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, in_channels, 3, padding=1)
        )

    def forward(self, x, t):
        # x: (B, C, H, W), t: (B,)
        # time embedding
        t_emb = sinusoidal_time_emb(t, self.time_emb.mlp[0].in_features)
        t_emb = self.time_emb(t_emb)

        # initial
        h = self.conv_in(x)
        hs = []

        # down-sampling
        idx = 0
        for ds in self.downsamples:
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[idx](h, t_emb)
                hs.append(h)
                idx += 1
            h = ds(h)

        # bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # up-sampling
        idx = 0
        for us in self.upsamples:
            h = us(h)
            for _ in range(self.num_res_blocks):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.up_blocks[idx](h, t_emb)
                idx += 1

        return self.conv_out(h)
