import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class MultiLabelDataset(Dataset):
    """
    A PyTorch Dataset for multi-label image data.
    Each sample returns:
      - image tensor
      - a tensor of label indices
    """
    def __init__(self,
                 annotations_path: str,
                 image_dir: str,
                 label_map_path: str,
                 transform=None):
        # Load filename-to-labels mapping
        with open(annotations_path, 'r') as f:
            self.data = json.load(f)
        # Load label-to-index mapping
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
        self.image_dir = image_dir
        self.transform = transform
        # List of all filenames
        self.filenames = list(self.data.keys())
        # Number of distinct labels
        self.num_labels = len(self.label_map)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Get filename and labels
        fname = self.filenames[idx]
        labels = self.data[fname]  # e.g. ["cyan cube", "brown cylinder"]

        # Load and transform image
        img_path = os.path.join(self.image_dir, fname)
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # Convert label names to indices
        label_indices = [self.label_map[l] for l in labels]
        label_tensor = torch.tensor(label_indices, dtype=torch.long)

        return image, label_tensor


def multi_label_collate(batch):
    """
    Custom collate function that stacks images and keeps
    a list of label-index tensors for embedding later.

    Returns:
      - images: FloatTensor of shape (batch_size, C, H, W)
      - label_lists: list of 1D LongTensors (variable lengths)
    """
    images, label_lists = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(label_lists)


# Example usage:
if __name__ == "__main__":
    from torchvision import transforms
    # Transform pipeline (include normalization)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = MultiLabelDataset(
        annotations_path="../file/train.json",
        image_dir="../iclevr",
        label_map_path="../label_map.json",
        transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=multi_label_collate,
        num_workers=4
    )

    # Iterate one batch to verify
    imgs, label_lists = next(iter(loader))
    print(imgs.shape)             # e.g. (32, 3, 64, 64)
    print(label_lists[0])         # e.g. tensor([3, 7])
    print(len(label_lists))       # 32
