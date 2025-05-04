import os
import math
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataloader import MultiLabelDataset, multi_label_collate
from src.model import AttentionUNet
from torch.optim import Adam
from tqdm import trange, tqdm


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, timesteps)


class DiffusionTrainer:
    def __init__(
        self,
        data_json,
        image_dir,
        label_map_path,
        checkpoint_dir,
        epochs=50,
        batch_size=32,
        lr=2e-4,
        img_size=64,
        timesteps=1000,
        cond_emb_dim=256,
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
        # Model
        self.model = AttentionUNet(
            in_channels=3,
            base_channels=64,
            channel_mults=(1,2,4,8),
            time_emb_dim=256,
            cond_emb_dim=cond_emb_dim,
            out_channels=3
        ).to(self.device)
        # Optimizer
        self.opt = Adam(
            list(self.model.parameters()) + list(self.cond_emb.parameters()), lr=lr
        )
        # Diffusion schedule
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps).to(self.device)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(self.device)
        # Training settings
        self.epochs = epochs
        # Checkpoint dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.ckpt_dir = checkpoint_dir

    def train(self):
        mse = nn.MSELoss()
        # Outer loop with epoch progress
        for epoch in trange(1, self.epochs + 1, desc="Epochs", unit="epoch"):
            self.model.train()
            total_loss = 0.0
            # Inner loop with batch progress
            for imgs, label_lists in tqdm(self.loader, desc=f"Epoch {epoch}/{self.epochs}", unit="batch", leave=False):
                B = imgs.size(0)
                imgs = imgs.to(self.device)
                # Build cond embeddings
                flat = torch.cat(label_lists)
                batch_idx = []
                for i, lbls in enumerate(label_lists):
                    batch_idx += [i] * len(lbls)
                lbls = flat.to(self.device)
                idx = torch.tensor(batch_idx, device=self.device)
                c_emb = self.cond_emb(lbls)
                c_emb = torch.zeros(B, c_emb.size(-1), device=self.device).index_add_(0, idx, c_emb)
                # sample t
                t = torch.randint(0, self.timesteps, (B,), device=self.device).long()
                # noise
                noise = torch.randn_like(imgs)
                sqrt_acp = self.alphas_cumprod[t].sqrt().view(B,1,1,1)
                sqrt_1_acp = (1 - self.alphas_cumprod[t]).sqrt().view(B,1,1,1)
                xt = sqrt_acp * imgs + sqrt_1_acp * noise
                # predict
                pred = self.model(xt, t, c_emb)
                loss = mse(pred, noise)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                total_loss += loss.item()
            # Average loss for epoch
            avg_loss = total_loss / len(self.loader)
            tqdm.write(f"Epoch {epoch}/{self.epochs} - Loss: {avg_loss:.4f}")
            # save checkpoint
            torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, f"model_epoch{epoch}.pth"))


if __name__ == "__main__":
    # Paths
    train_json = "./file/train.json"
    image_dir = "./iclevr"
    label_map_path = "label_map.json"
    ckpt_dir = "./checkpoints"
    # Hyperparams
    epochs = 50
    batch_size = 4
    lr = 2e-4
    img_size = 64
    timesteps = 1000

    trainer = DiffusionTrainer(
        train_json,
        image_dir,
        label_map_path,
        ckpt_dir,
        epochs,
        batch_size,
        lr,
        img_size,
        timesteps,
        cond_emb_dim=256
    )
    trainer.train()
