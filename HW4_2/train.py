from src.trainer import DiffusionTrainer


if __name__ == "__main__":
    # Paths & hyperparameters
    train_json     = "./file/train.json"
    image_dir      = "./iclevr"
    label_map_path = "label_map.json"
    ckpt_dir       = "./checkpoints_hq/"
    epochs         = 500
    batch_size     = 8
    lr             = 5e-5
    img_size       = 128
    timesteps      = 4000
    cond_emb_dim   = 512
    base_channels  = 128

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
    trainer.train()
