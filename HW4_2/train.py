import os
import argparse
from src.trainer import DiffusionTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a .pt checkpoint to resume from")
    args = parser.parse_args()

    # Paths & hyperparameters
    train_json     = "./file/train.json"
    image_dir      = "./iclevr"
    label_map_path = "label_map.json"
    ckpt_dir       = "./checkpoints_hq/"
    epochs         = 500
    batch_size     = 4
    lr             = 5e-5
    img_size       = 64
    timesteps      = 4000
    cond_emb_dim   = 512
    base_channels  = 64

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
        cond_emb_dim,
        base_channels
    )

    # If you supplied --resume_from, pass it here; otherwise starts fresh
    trainer.train(resume_from=args.resume_from)
