import torch
from torch.utils.data import DataLoader
from src.models.unet import Unet
from src.models.resnet34_unetv2 import ResUnet_CBAM
from src.oxford_pet import Compose, MinMaxNormalization, ToTensor, SimpleOxfordPetDataset
from src.utils import soft_dice_loss
from src.inference import get_args, pred_output_t_mask
from tqdm import tqdm

def get_test_loader(data_path, batch_size, transform):
    dataset = SimpleOxfordPetDataset(root=data_path, mode="test", transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset

def load_model(model_class, checkpoint_path, device, **kwargs):
    model = model_class(**kwargs).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def evaluate_model(model, dataloader, device):
    total_score = 0.0
    total_samples = 0
    with torch.no_grad():
        with tqdm(dataloader, leave=False) as pbar:
            for batch in pbar:
                inputs = batch["image"].float().to(device)
                targets = batch["mask"].float().to(device)

                if targets.dim() == 3:
                    targets = targets.unsqueeze(1)
                    
                # Convert target masks to binary using a threshold of 0.5.
                targets = (targets > 0.5).float()

                outputs = model(inputs)
                outputs_mask = pred_output_t_mask(outputs)
                
                # soft_dice_loss returns (loss, dice_score) where dice_score is averaged over the batch.
                # Multiply the average dice score by the batch size to get the sum of dice scores for that batch.
                batch_score_avg = soft_dice_loss(outputs_mask, targets)[1]
                batch_size = inputs.size(0)
                total_score += batch_score_avg * batch_size
                total_samples += batch_size

    # Compute the overall average dice score over all samples.
    return total_score / total_samples if total_samples > 0 else 0.0


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Transforms ---
    transform_basic = Compose([MinMaxNormalization(), ToTensor()])
    transform_tensor = ToTensor()

    # --- Loaders ---
    loader_resunet, _ = get_test_loader(args.data_path, args.batch_size, transform_basic)
    loader_unet, _ = get_test_loader(args.data_path, args.batch_size, transform_tensor)

    # --- Model Paths ---
    # Uncomment and set the appropriate checkpoint paths as needed:
    resunet_ckpt = "C:/Users/daikon/jimmy/DL/HW2_CNN/runs/checkpoints/laset_try_RESunetAttention_basic_Trans_NFFT_w_schedu/best_checkpoint.pth"
    # unet_ckpt = "path/to/unet_checkpoint.pth"
    unet_ckpt = "C:/Users/daikon/jimmy/DL/HW2_CNN/runs/checkpoints/RESUME_Unet_wo_trans2/best_checkpoint.pth"

    # --- Load Models ---
    # For example, if you want to evaluate the UNet model:
    unet = load_model(Unet, unet_ckpt, device, n_channels=3, n_classes=1)
    # To evaluate the ResUnet model, uncomment the next lines:
    resunet = load_model(ResUnet_CBAM, resunet_ckpt, device, in_channels=3, out_channels=1)

    # --- Evaluate Models ---
    # Evaluate UNet with no additional transform
    print("Evaluating UNet + No Transform...")
    unet_score = evaluate_model(unet, loader_unet, device)
    print(f"UNet Dice Score: {unet_score:.4f}")

    # Example for evaluating ResUnet with basic transform (if desired):
    print("Evaluating ResUnet + Basic Transform...")
    resunet_score = evaluate_model(resunet, loader_resunet, device)
    print(f"ResUnet Dice Score: {resunet_score:.4f}")

if __name__ == "__main__":
    main()
