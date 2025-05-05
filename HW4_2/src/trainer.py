import os
import math
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataloader import MultiLabelDataset, multi_label_collate
from src.model import AttentionUNet
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange, tqdm


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in Nichol & Dhariwal (Improved DDPM).
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999)


def get_scheduler(optimizer, total_steps, warmup_ratio=0.1):
    warmup_steps = int(warmup_ratio * total_steps)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


class DiffusionTrainer:
    def __init__(
        self,
        data_json,
        image_dir,
        label_map_path,
        checkpoint_dir,
        epochs=500,
        batch_size=8,
        lr=5e-5,
        img_size=128,
        timesteps=4000,
        cond_emb_dim=512,
        base_channels=128,
        device=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Data
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.dataset = MultiLabelDataset(data_json, image_dir, label_map_path, transform)
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=multi_label_collate,
            num_workers=4,
            pin_memory=True
        )

        # Label embedding
        self.num_labels = self.dataset.num_labels
        self.cond_emb = nn.Embedding(self.num_labels, cond_emb_dim).to(self.device)

        # Model + EMA copy
        self.model = AttentionUNet(
            in_channels=3,
            base_channels=base_channels,
            channel_mults=(1, 2, 4, 8),
            time_emb_dim=256,
            cond_emb_dim=cond_emb_dim,
            out_channels=3
        ).to(self.device)
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        self.ema_decay = 0.99995

        # Optimizer
        self.opt = Adam(
            list(self.model.parameters()) + list(self.cond_emb.parameters()),
            lr=lr
        )

        # Scheduler: warmup + cosine decay
        num_batches = math.ceil(len(self.dataset) / batch_size)
        total_steps = epochs * num_batches // 4  # effective optimizer steps because of grad-accumulation
        self.scheduler = get_scheduler(self.opt, total_steps)

        # Diffusion schedule
        self.timesteps = timesteps
        self.betas = cosine_beta_schedule(timesteps).to(self.device)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(self.device)

        # Training config
        self.epochs = epochs
        self.accumulate_steps = 2
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.ckpt_dir = checkpoint_dir
        self.global_step = 0

    def save_checkpoint(self, epoch):
        path = os.path.join(self.ckpt_dir, f"hq_ckpt_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "embed": self.cond_emb.state_dict(),
            "opt": self.opt.state_dict(),
            "sched": self.scheduler.state_dict()
        }, path)

    def train(self):
        mse = nn.MSELoss()

        for epoch in trange(1, self.epochs + 1, desc="Epochs", unit="epoch"):
            self.model.train()
            total_loss = 0.0

            for step, (imgs, label_lists) in enumerate(tqdm(self.loader, desc=f"Epoch {epoch}", leave=False)):
                B = imgs.size(0)
                imgs = imgs.to(self.device)

                # Build conditional embedding
                flat = torch.cat(label_lists)
                batch_idx = []
                for i, lbls in enumerate(label_lists):
                    batch_idx += [i] * len(lbls)
                lbls = flat.to(self.device)
                idx = torch.tensor(batch_idx, device=self.device)
                c_emb = self.cond_emb(lbls)
                c_emb = torch.zeros(B, c_emb.size(-1), device=self.device).index_add_(0, idx, c_emb)

                # Sample t and noise
                t = torch.randint(0, self.timesteps, (B,), device=self.device).long()
                noise = torch.randn_like(imgs)
                sqrt_acp = self.alphas_cumprod[t].sqrt().view(B,1,1,1)
                sqrt_1_acp = (1 - self.alphas_cumprod[t]).sqrt().view(B,1,1,1)
                xt = sqrt_acp * imgs + sqrt_1_acp * noise

                # Predict
                pred = self.model(xt, t, c_emb)
                orig_loss = mse(pred, noise)
                loss = orig_loss / self.accumulate_steps
                loss.backward()

                # Gradient accumulation / optimizer step
                if (step + 1) % self.accumulate_steps == 0:
                    # Clip gradients
                    nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.cond_emb.parameters()),
                        max_norm=1.0
                    )
                    self.opt.step()
                    self.scheduler.step()
                    self.opt.zero_grad()

                    # EMA update
                    with torch.no_grad():
                        for p, p_ema in zip(self.model.parameters(), self.ema_model.parameters()):
                            p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

                    self.global_step += 1

                total_loss += orig_loss.item()

            avg_loss = total_loss / len(self.loader)
            tqdm.write(f"Epoch {epoch}/{self.epochs} - Avg Loss: {avg_loss:.4f} - LR: {self.opt.param_groups[0]['lr']:.2e}")
            self.save_checkpoint(epoch)