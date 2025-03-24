import os
import torch
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        sample = dict(image=image, mask=mask, trimap=trimap)
        # if self.transform is not None:
        #     sample = self.transform(**sample)
        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):
        # Get the raw sample from the base class (with numpy arrays)
        sample = super().__getitem__(*args, **kwargs)

        # Resize the images using PIL while the data is still in numpy format.
        image = np.array(
            Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR)
        )
        mask = np.array(
            Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST)
        )
        trimap = np.array(
            Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST)
        )

        # Resize the images using PIL while the data is still in numpy format.
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        sample["image"] = image
        sample["mask"] = mask
        sample["trimap"] = trimap


        # Now apply additional transforms (which may include ToTensor, normalization, etc.)
        if self.transform is not None:
            sample = self.transform(**sample)
        return sample
    

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)

def load_dataset(data_path, mode, batch_size=32, shuffle=True, num_workers=4, transform=None):
    # Create an instance of your dataset. Here we use the SimpleOxfordPetDataset.
    dataset = SimpleOxfordPetDataset(root=data_path, mode=mode, transform=transform)

    # Create a DataLoader from the dataset.
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )
    
    return dataloader


# -------------------------------------------------
# Transformation classes for binary segmentation tasks
# -------------------------------------------------

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **sample):
        for t in self.transforms:
            sample = t(**sample)
        return sample

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, **sample):
        if np.random.rand() < self.p:
            sample["image"] = np.fliplr(sample["image"]).copy()
            sample["mask"] = np.fliplr(sample["mask"]).copy()
            sample["trimap"] = np.fliplr(sample["trimap"]).copy()
        return sample

class RandomRotation90:
    def __call__(self, **sample):
        k = np.random.randint(0, 4)
        if k:
            sample["image"] = np.rot90(sample["image"], k)
            sample["mask"] = np.rot90(sample["mask"], k)
            sample["trimap"] = np.rot90(sample["trimap"], k)
        return sample

class MinMaxNormalization:
    def __call__(self, **sample):
        image = sample["image"]
        image = image.astype(np.float32) / 255.0
        sample["image"] = image
        return sample

class ToTensor:
    def __call__(self, **sample):
        sample["image"] = torch.from_numpy(sample["image"].copy()).float().permute(2, 0, 1) # CHW here
        sample["mask"] = torch.from_numpy(sample["mask"].copy()).float()
        sample["trimap"] = torch.from_numpy(sample["trimap"].copy()).float()
        return sample

