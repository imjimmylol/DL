import os
import math
import yaml
import torch
import argparse
from torchvision.utils import save_image
from src.model import UNet


def cosine_beta_schedule(T, s=0.008):
    """
    Cosine schedule as used in Nichol & Dhariwal (2021).
    """
    steps = T + 1
    t = torch.linspace(0, T, steps)
    f = torch.cos(((t / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = f / f[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(0, 0.999)


def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def sample_images(model, device, betas, num_samples, img_size):
    T = betas.size(0)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.], device=device), alphas_cumprod[:-1]], dim=0)

    # Prepare noise
    x = torch.randn((num_samples, model.in_channels, img_size, img_size), device=device)

    for t in reversed(range(T)):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        # Predict noise
        with torch.no_grad():
            eps_pred = model(x, t_batch)

        # Compute posterior mean and variance
        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]
        alpha_bar_prev = alphas_cumprod_prev[t]
        beta_t = betas[t]

        # Coefficients
        coef1 = beta_t * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar_t)
        coef2 = (1 - alpha_bar_prev) * torch.sqrt(alpha_t) / (1 - alpha_bar_t)
        posterior_mean = coef1.view(-1, 1, 1, 1) * (x - torch.sqrt(1 - alpha_bar_t).view(-1, 1, 1, 1) * eps_pred) + coef2.view(-1, 1, 1, 1) * x

        # Add noise except for t == 0
        if t > 0:
            noise = torch.randn_like(x)
            posterior_var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
            x = posterior_mean + torch.sqrt(posterior_var).view(-1, 1, 1, 1) * noise
        else:
            x = posterior_mean

    # Scale back to [0, 1]
    x = (x * 0.5) + 0.5
    return x


def main():
    parser = argparse.ArgumentParser(description="Diffusion Model Inference")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config file.")
    parser.add_argument("--checkpoint", "-ckpt", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--output_dir", "-o", default="samples", help="Directory to save generated images.")
    parser.add_argument("--num_samples", "-n", type=int, default=8, help="Number of samples to generate.")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    img_size = cfg.get("img_size", 128)

    # Initialize model
    model = UNet(
        in_channels=cfg["in_channels"],
        base_channels=cfg["base_channels"],
        time_emb_dim=cfg["time_emb_dim"]
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Prepare diffusion schedule
    T = int(cfg["T"])
    betas = cosine_beta_schedule(T).to(device)

    # Sample images
    samples = sample_images(model, device, betas, args.num_samples, img_size)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save individual images
    for i, img in enumerate(samples):
        save_path = os.path.join(args.output_dir, f"sample_{i:03d}.png")
        save_image(img, save_path)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
