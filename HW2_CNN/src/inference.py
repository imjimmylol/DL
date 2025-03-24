import argparse
import torch
from src.models.unet import Unet
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from src.oxford_pet import *
from src.models.unet import Unet 
from src.models.resnet34_unetv2 import ResUnet_CBAM
from src.utils import soft_dice_loss
import os

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    # parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    # parser.add_argument('--transform', 't', type=bool, default=None)
    return parser.parse_args()

def pred_output_t_mask(output):
    outputs_resized = F.interpolate(output, size=(256, 256), mode='bilinear', align_corners=False)
    outputs_prob = torch.sigmoid(outputs_resized)
    # pred_mask = (outputs_prob > 0.5).float().cpu().numpy().squeeze()
    pred_mask = (outputs_prob > 0.5).float()
    return pred_mask



# if __name__ == '__main__':
#     args = get_args()

#     test_basic_transform = Compose([
#         MinMaxNormalization(),
#         ToTensor()
#     ])

#     testset_w_trans = SimpleOxfordPetDataset(root=args.data_path, mode="test", 
#                                      transform=test_basic_transform)
#     testset_wo_trans = SimpleOxfordPetDataset(root=args.data_path, mode="test", 
#                                      transform=ToTensor())
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#     resu_w_trans = ResUnet_CBAM(in_channels=3, out_channels=1).to(device)
#     unet_wo_trans = Unet(n_channels=3, n_classes=1).to(device)

#     checkpoint_resu_w_trans = torch.load("C:/Users/daikon/jimmy/DL/HW2_CNN/runs/checkpoints/resu_w_basic_trans/best_checkpoint.pth", map_location=device)
#     checkpoint_unet_wo_trans = torch.load("C:/Users/daikon/jimmy/DL/HW2_CNN/runs/checkpoints/fix/best_checkpoint.pth", map_location=device)
    
#     resu_w_trans.load_state_dict(checkpoint_resu_w_trans['model_state_dict'])
#     unet_wo_trans.load_state_dict(checkpoint_unet_wo_trans['model_state_dict'])
    
#     resu_w_trans.eval()
#     unet_wo_trans.eval()