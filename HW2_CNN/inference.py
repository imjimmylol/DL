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


def evaluate_model(model, dataloader, dataset_len, device):
    total_score = 0
    with tqdm(dataloader, leave=False) as pbar:
        for batch in pbar:
            inputs = batch["image"].float().to(device)
            targets = batch["mask"].float().to(device)

            if targets.dim() == 3:
                targets = targets.unsqueeze(1)

            outputs = model(inputs)
            outputs_mask = pred_output_t_mask(outputs)

            score = soft_dice_loss(outputs_mask, targets.cpu().detach().numpy())[1]
            total_score += score
    return total_score / dataset_len


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Transforms ---
    transform_basic = Compose([MinMaxNormalization(), ToTensor()])
    transform_tensor = ToTensor()

    # --- Loaders ---
    loader_resunet, dataset_resunet = get_test_loader(args.data_path, args.batch_size, transform_basic)
    loader_unet, dataset_unet = get_test_loader(args.data_path, args.batch_size, transform_tensor)

    # --- Model Paths ---
    resunet_ckpt = "C:/Users/daikon/jimmy/DL/HW2_CNN/runs/checkpoints/resu_w_basic_trans/best_checkpoint.pth"
    unet_ckpt = "C:/Users/daikon/jimmy/DL/HW2_CNN/runs/checkpoints/fix/best_checkpoint.pth"

    # --- Load Models ---
    resunet = load_model(ResUnet_CBAM, resunet_ckpt, device, in_channels=3, out_channels=1)
    unet = load_model(Unet, unet_ckpt, device, n_channels=3, n_classes=1)

    # --- Evaluate Models ---
    print("Evaluating ResUnet + Basic Transform...")
    resunet_score = evaluate_model(resunet, loader_resunet, len(dataset_resunet), device)
    print(f"ResUnet Dice Score: {resunet_score:.4f}")

    print("Evaluating UNet + No Transform...")
    unet_score = evaluate_model(unet, loader_unet, len(dataset_unet), device)
    print(f"UNet Dice Score: {unet_score:.4f}")


if __name__ == "__main__":
    main()
