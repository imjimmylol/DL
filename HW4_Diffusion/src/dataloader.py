import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class ClevrDataset(Dataset):
    def __init__(self,
                 index_file: str,
                 data_dir: str = "./iclevr",
                 transform=None,
                 encode_labels: bool = True):
        """
        index_file: path to JSON, either
          • dict {fname: [labels], …}  → training/val with images
          • list [[labels], …]         → test, label‐only
        """
        with open(index_file, 'r') as f:
            loaded = json.load(f)

        # Are we in "label‐only" mode?
        if isinstance(loaded, dict):
            self.items      = list(loaded.items())   # [(fname, [labels]), …]
            self.label_only = False
        elif isinstance(loaded, list):
            # no filenames, just index them
            self.items      = [(str(i), labels) for i, labels in enumerate(loaded)]
            self.label_only = True
        else:
            raise ValueError("JSON must be dict or list of label‐lists")

        self.data_dir      = data_dir
        self.transform     = transform
        self.encode_labels = encode_labels

        if self.encode_labels:
            # build vocab from all labels in items
            all_labels = {lbl for _, labels in self.items for lbl in labels}
            self.classes     = sorted(all_labels)
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, labels = self.items[idx]

        # 1) If test / label‐only, just return labels or embedding:
        if self.label_only:
            if not self.encode_labels:
                return labels
            # build multi-hot
            target = torch.zeros(len(self.classes), dtype=torch.float32)
            for lbl in labels:
                target[self.class_to_idx[lbl]] = 1.0
            return target

        # 2) Otherwise (training/val), load image + labels:
        img_path = os.path.join(self.data_dir, fname)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if not self.encode_labels:
            return img, labels

        target = torch.zeros(len(self.classes), dtype=torch.float32)
        for lbl in labels:
            target[self.class_to_idx[lbl]] = 1.0

        return img, target



# # example usage:

# # 1. define any image transforms you like
# transform = transforms.Compose([
#     transforms.Resize((128,128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5,0.5,0.5],
#                          std=[0.5,0.5,0.5])
# ])

# # 2. instantiate the dataset
# dataset = ClevrDataset(
#     index_file="train_index.json",
#     data_dir="./iclevr",
#     transform=transform,
#     encode_labels=True   # set False if you just want the raw list of strings
# )

# # 3. wrap in a DataLoader
# dataloader = DataLoader(
#     dataset,
#     batch_size=32,
#     shuffle=True,
#     num_workers=4,
#     pin_memory=True
# )

# # 4. iterate!
# for imgs, targets in dataloader:
#     # imgs:    [32, 3, 128, 128] tensor
#     # targets: [32, num_classes] multi-hot tensor
#     ...
