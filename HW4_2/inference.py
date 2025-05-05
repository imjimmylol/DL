# inference.py

import os
import json
import torch
import argparse
from torchvision.utils import save_image
from src.model import AttentionUNet   # your UNet implementation
import torch.nn as nn
from tqdm import tqdm

def make_beta_schedule(T, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, T)


@torch.no_grad()
def sample_ddpm_batch(model, embed, label_map, conditions, out_dir,
                      img_size=64, T=1000, beta_start=1e-4, beta_end=0.02,
                      batch_size=2):
    device = next(model.parameters()).device
    betas = make_beta_schedule(T, beta_start, beta_end).to(device)
    alphas = 1 - betas
    alphas_cum = torch.cumprod(alphas, dim=0)

    os.makedirs(out_dir, exist_ok=True)
    num_conds = len(conditions)

    for start in range(0, num_conds, batch_size):
        batch_conds = conditions[start:start + batch_size]
        B = len(batch_conds)

        # build batched condition embeddings
        emb_list = []
        for labels in batch_conds:
            idxs = torch.LongTensor([label_map[l] for l in labels]).to(device)
            emb = embed(idxs)            # (num_labels_i, cond_emb_dim)
            emb_sum = emb.sum(dim=0)     # (cond_emb_dim,)
            emb_list.append(emb_sum)
        cond_emb_batch = torch.stack(emb_list, dim=0)  # (B, cond_emb_dim)

        # start from noise for all in batch
        x = torch.randn(B, 3, img_size, img_size, device=device)
        # print(43)
        # reverse diffusion loop
        for t in tqdm(reversed(range(T))):
            t_vec = torch.full((B,), t, device=device, dtype=torch.long)
            eps_pred = model(x, t_vec, cond_emb_batch)

            a_t = alphas[t]
            a_cum = alphas_cum[t]
            a_prev = alphas_cum[t - 1] if t > 0 else torch.tensor(1.0, device=device)

            coef1 = 1.0 / torch.sqrt(a_t)
            coef2 = (1 - a_t) / torch.sqrt(1 - a_cum)
            mean = coef1 * (x - coef2 * eps_pred)

            if t > 0:
                var = betas[t] * (1 - a_prev) / (1 - a_cum)
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(var) * noise
            else:
                x = mean
        # print(63)
        # denormalize & save
        imgs = (x.clamp(-1, 1) + 1) * 0.5  # map from [-1,1] to [0,1]
        for j, labels in enumerate(batch_conds):
            idx = start + j
            filename = os.path.join(out_dir, f"{idx:03d}.png")
            save_image(imgs[j], filename)
            print(f"[{idx+1}/{num_conds}] Saved: {labels} -> {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./checkpoints_hq/hq_ckpt_epoch60.pt",
                        help="Path to trained checkpoint")
    parser.add_argument("--label_map", default="label_map.json",
                        help="Path to label_map.json")
    parser.add_argument("--cond_json", default="./file/test.json",
                        help="Path to JSON file containing list of label-sets")
    parser.add_argument("--out_dir", default="outputs",
                        help="Directory for generated images")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of samples to generate at once")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load label map
    with open(args.label_map, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    num_labels = len(label_map)
    cond_emb_dim  = 512    # <— must match your trainer!
    base_channels = 64    # <— must match your trainer!
    time_emb_dim  = 256

    unet = AttentionUNet(
        in_channels=3,
        base_channels=base_channels,
        channel_mults=(1,2,4,8),
        time_emb_dim=time_emb_dim,
        cond_emb_dim=cond_emb_dim,
        out_channels=3
    ).to(device)

    # Optionally, if you want to use your EMA model at inference time:
    ema_unet = AttentionUNet(
        in_channels=3,
        base_channels=base_channels,
        channel_mults=(1,2,4,8),
        time_emb_dim=time_emb_dim,
        cond_emb_dim=cond_emb_dim,
        out_channels=3
    ).to(device)

    embed = nn.Embedding(num_labels, cond_emb_dim).to(device)
    # load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    unet.load_state_dict(
        ckpt.get("model", ckpt)
    )
    embed.load_state_dict(
        ckpt.get("embed", embed.state_dict())
    )
    unet.eval(); embed.eval()

    # load conditions from JSON
    with open(args.cond_json, 'r', encoding='utf-8') as f:
        conditions = json.load(f)
        # expect: [["red cube"], ["blue sphere","green cube"], ...]

    sample_ddpm_batch(
        model=unet,
        embed=embed,
        label_map=label_map,
        conditions=conditions,
        out_dir=args.out_dir,
        img_size=args.img_size,
        T=args.T,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
