import torch
from torchvision import transforms


class Denormalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
    def forward(self, tensor):
        # tensor is [B, C, H, W] in [-1,1]
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor.clamp(0, 1)   # now in [0,1]


# eval_transform = transforms.Compose([
#     Denormalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
#     transforms.Resize((64,64)),
#     transforms.Normalize((0.5,0.5,0.5),
#                          (0.5,0.5,0.5)),
# ])
