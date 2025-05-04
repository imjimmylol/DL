import os
import math
import yaml
import wandb
import torch
import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataloader import ClevrDataset
from src.model import UNet
from tqdm import tqdm  


def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    t = torch.linspace(0, T, steps)
    f = torch.cos(((t / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = f / f[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(0, 0.999)


def collate_ignore_labels(batch):
    imgs = torch.stack([item[0] for item in batch], dim=0)
    return imgs, None


def save_checkpoint(state, ckpt_dir, epoch):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch}.pth")
    torch.save(state, path)
    print(f"=> Saved checkpoint: {path}")


def train_diffusion(cfg):
    # --- 1) Initialize W&B
    wandb.init(project=cfg["project_name"], config=cfg)
    config = wandb.config

    # --- 2) Data loader
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    dataset = ClevrDataset(
        index_file=cfg["index_file"],
        data_dir=cfg["data_dir"],
        transform=transform,
        encode_labels=False
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_ignore_labels
    )

    # --- 3) Model + optimizer
    device = torch.device(config.device)
    model = UNet(
        in_channels=config.in_channels,
        base_channels=config.base_channels,
        time_emb_dim=config.time_emb_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    wandb.watch(model, log="all", log_freq=100)

    # --- 4) Noise schedule
    betas = cosine_beta_schedule(config.T).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # --- 5) Resume logic
    start_epoch = 1
    global_step = 0
    resume_path = cfg.get("resume_checkpoint")
    if resume_path:
        if os.path.isfile(resume_path):
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            global_step = ckpt.get("global_step", 0)
            print(f"=> Resumed from checkpoint '{resume_path}' (starting at epoch {start_epoch})")
        else:
            print(f"[!] No checkpoint found at '{resume_path}' — starting from scratch.")

    # --- 6) Training loop
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{config.epochs}", unit="batch", leave=False)

        for imgs, _ in pbar:
            imgs = imgs.to(device)
            # sample t, noise, form x_t
            t = torch.randint(0, config.T, (imgs.size(0),), device=device).long()
            noise = torch.randn_like(imgs)
            sqrt_ab = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
            sqrt_1_ab = (1 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
            x_t = sqrt_ab * imgs + sqrt_1_ab * noise

            # predict and step
            eps_pred = model(x_t, t)
            loss = F.mse_loss(eps_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging
            running_loss += loss.item() * imgs.size(0)
            global_step += 1
            wandb.log({"train/loss": loss.item()}, step=global_step)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(dataset)
        print(f"[Epoch {epoch}/{config.epochs}] loss: {avg_loss:.4f}")
        wandb.log({"epoch/loss": avg_loss, "epoch": epoch}, step=global_step)

        # --- 7) Checkpointing
        if epoch % config.checkpoint_interval == 0:
            ckpt_state = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            save_checkpoint(ckpt_state, config.checkpoint_dir, epoch)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to YAML config file."
    )
    args = parser.parse_args()

    # load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        cfg["lr"]          = float(cfg["lr"])
        cfg["batch_size"]  = int(cfg["batch_size"])
        cfg["epochs"]      = int(cfg["epochs"])
        cfg["T"]           = int(cfg["T"])        

    train_diffusion(cfg)
