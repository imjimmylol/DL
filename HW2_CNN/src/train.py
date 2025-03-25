import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.oxford_pet import *
from src.models.unet import Unet 
from src.models.resnet34_unetv2 import ResUnet_CBAM
from src.models.resnet34_unetv3 import ResNetUNet
from src.utils import TensorboardLogger
from torch.utils.data import Dataset, DataLoader

def train(args, train_transform=None, valid_transform=None):
    trainset = SimpleOxfordPetDataset(root="./data", mode="train", transform=train_transform if train_transform is not None else ToTensor())
    validset = SimpleOxfordPetDataset(root="./data", mode="valid", transform=valid_transform if valid_transform is not None else ToTensor())

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    validloader = DataLoader(validset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = Unet(n_channels=3, n_classes=1).to(device)
    # model = ResUnet_CBAM(in_channels=3, out_channels=1).to(device)
    model = ResNetUNet(n_classes=1).to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Add scheduler (ReduceLROnPlateau)
    scheduler = None
    if getattr(args, "use_scheduler", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

    logger = TensorboardLogger(log_dir=args.log_dir, run_name=args.runame)

    checkpoint_dir = os.path.join(args.log_dir, "checkpoints", args.runame)
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0
    best_valid_loss = float("inf")
    if hasattr(args, "resume") and args.resume is not None:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_valid_loss = checkpoint.get('valid_loss', best_valid_loss)
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at '{args.resume}'")

    num_epochs = args.epochs

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0

        with tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False) as pbar:
            for batch in pbar:
                inputs = batch["image"].float().to(device)
                targets = batch["mask"].float().to(device)

                optimizer.zero_grad()
                if targets.dim() == 3:
                    targets = targets.unsqueeze(1)

                outputs = model(inputs)
                outputs_resized = F.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=False)
                outputs_prob = torch.sigmoid(outputs_resized)

                loss = criterion(outputs_prob, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(trainloader.dataset)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            with tqdm(validloader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=False) as pbar:
                for batch in pbar:
                    inputs = batch["image"].float().to(device)
                    targets = batch["mask"].float().to(device)
                    if targets.dim() == 3:
                        targets = targets.unsqueeze(1)
                    outputs = model(inputs)
                    outputs_resized = F.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=False)
                    outputs_prob = torch.sigmoid(outputs_resized)

                    loss = criterion(outputs_prob, targets)
                    valid_loss += loss.item() * inputs.size(0)
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

        valid_loss /= len(validloader.dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

        logger.log_loss(train_loss, valid_loss, epoch)

        # Step scheduler if available
        if scheduler is not None:
            scheduler.step(valid_loss)

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss,
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
            torch.save(checkpoint, best_checkpoint_path)
            print(f"Saved best checkpoint: {best_checkpoint_path}")

        if epoch % args.log_interval == 0:
            batch = next(iter(validloader))
            inputs = batch["image"].float().to(device)
            targets = batch["mask"].float().to(device)

            outputs = model(inputs)
            outputs_resized = F.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=False)
            outputs_prob = torch.sigmoid(outputs_resized)

            input_img = inputs[0].cpu().numpy().transpose(1, 2, 0)
            target_mask = targets[0].cpu().numpy().squeeze()
            pred_mask = (outputs_prob[0] > 0.5).float().cpu().numpy().squeeze()

            fig = TensorboardLogger.create_comparison_figure(input_img, target_mask, pred_mask)
            logger.log_figure("Predictions", fig, epoch)
            plt.close(fig)

    logger.close()
