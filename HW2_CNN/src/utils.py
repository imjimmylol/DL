import torch
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter

def soft_dice_loss(preds, targets, eps=1e-6):
    # Assuming shape (B, 1, H, W)
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean(), dice.mean()


class TensorboardLogger:
    def __init__(self, log_dir, run_name="default_run"):
        # Create a unique logging directory by appending run_name
        run_log_dir = os.path.join(log_dir, run_name)
        self.writer = SummaryWriter(log_dir=run_log_dir)
    
    def log_loss(self, train_loss, valid_loss, epoch):
        self.writer.add_scalar("Loss/Train", train_loss, epoch)
        self.writer.add_scalar("Loss/Valid", valid_loss, epoch)
    
    def log_figure(self, tag, fig, epoch):
        self.writer.add_figure(tag, fig, epoch)
    
    @staticmethod
    def create_comparison_figure(input_img, target_mask, pred_mask):
        """
        Create a matplotlib figure that displays the input image,
        ground truth mask, and predicted mask side by side.
        """
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(input_img)
        ax[0].set_title("Input Image")
        ax[1].imshow(target_mask, cmap="gray")
        ax[1].set_title("Ground Truth")
        ax[2].imshow(pred_mask, cmap="gray")
        ax[2].set_title("Prediction")
        for a in ax:
            a.axis("off")
        return fig

    def close(self):
        self.writer.close()
