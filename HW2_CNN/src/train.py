import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.oxford_pet import *
from src.models.unet import Unet 
from src.models.resnet34_unetv2 import ResUnet_CBAM
from src.utils import TensorboardLogger
from torch.utils.data import Dataset, DataLoader

def train(args, train_transform=None, valid_transform=None):
    # 1. Load datasets for training and validation
    trainset = SimpleOxfordPetDataset(root="./data", mode="train", transform=train_transform if train_transform is not None else ToTensor())
    validset = SimpleOxfordPetDataset(root="./data", mode="valid", transform=valid_transform if valid_transform is not None else ToTensor())

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    validloader = DataLoader(validset, batch_size=args.batch_size, shuffle=True)

    # 2. Set device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Initialize the UNet model and move it to the appropriate device
    #    n_classes=1 for a single-channel (binary) output
    # model = Unet(n_channels=3, n_classes=1).to(device)
    model = ResUnet_CBAM(in_channels=3, out_channels=1).to(device)

    # 4. Define loss function (BCE Loss) and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 5. Initialize TensorBoard logger with run name
    # Here, we assume your TensorboardLogger accepts a "run_name" argument
    logger = TensorboardLogger(log_dir=args.log_dir, run_name=args.runame)
    
    # 6. Create checkpoint directory inside log_dir if it doesn't exist, including run name
    checkpoint_dir = os.path.join(args.log_dir, "checkpoints", args.runame)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Resume training if a checkpoint is provided ---
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
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at '{args.resume}'")

    num_epochs = args.epochs
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        
        # --- Training loop ---
        with tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False) as pbar:
            for batch in pbar:
                inputs = batch["image"].float().to(device)
                targets = batch["mask"].float().to(device)
                
                optimizer.zero_grad()
                
                # Forward pass: get raw logits from the model
                if targets.dim() == 3:
                    targets = targets.unsqueeze(1)

                outputs = model(inputs)
                # Resize the output to match target if needed
                outputs_resized = F.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=False)
                # Apply Sigmoid to obtain probabilities in [0,1]
                outputs_prob = torch.sigmoid(outputs_resized)
                
                # Compute BCE loss
                loss = criterion(outputs_prob, targets)
                
                # Backpropagation and optimization
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        train_loss /= len(trainloader.dataset)
        
        # --- Validation loop ---
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
        
        # Log losses using the logger
        logger.log_loss(train_loss, valid_loss, epoch)
        
        # --- Save checkpoints ---
        checkpoint = {
            'epoch': epoch + 1,  # epoch counter is incremented to indicate completion
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss,
        }
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save the best model checkpoint based on validation loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
            torch.save(checkpoint, best_checkpoint_path)
            print(f"Saved best checkpoint: {best_checkpoint_path}")
        
        # --- Log prediction outputs and comparisons every log_interval epochs ---
        if epoch % args.log_interval == 0:
            batch = next(iter(validloader))
            inputs = batch["image"].float().to(device)
            targets = batch["mask"].float().to(device)

            # Forward pass to get predictions
            outputs = model(inputs)
            outputs_resized = F.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=False)
            outputs_prob = torch.sigmoid(outputs_resized)

            # Convert predictions to a binary mask using threshold 0.5
            input_img = inputs[0].cpu().numpy().transpose(1, 2, 0)
            target_mask = targets[0].cpu().numpy().squeeze()
            pred_mask = (outputs_prob[0] > 0.5).float().cpu().numpy().squeeze()
            
            fig = TensorboardLogger.create_comparison_figure(input_img, target_mask, pred_mask)
            logger.log_figure("Predictions", fig, epoch)
            plt.close(fig)
    
    # Close the logger
    logger.close()
