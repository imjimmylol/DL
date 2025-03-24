import os
from src.oxford_pet import OxfordPetDataset  # Ensure the dataset class is in the same directory

# Set the root directory where the dataset will be stored
data_root = "./data"

# Create the dataset directory if it doesn't exist
os.makedirs(data_root, exist_ok=True)

# Download and extract the dataset
OxfordPetDataset.download(data_root)

print("Download and extraction complete!")
