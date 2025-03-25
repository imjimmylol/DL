import argparse
from src.train import train  # Import the train function
from src.oxford_pet import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="runs", help="TensorBoard log directory")
    parser.add_argument("--log_interval", type=int, default=1, help="Interval (in epochs) to log predictions")
    parser.add_argument("--data_path", type=str, default="./data", help="Path of the input data")
    parser.add_argument("--epochs", "-e", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=2, help="Batch size")
    parser.add_argument( "--lr", "-lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--runame", "-rn", type=str)
    parser.add_argument("--resume", default=None, help="dir of checkpt")
    parser.add_argument("--use_schedular", type=bool, default=False)
    args = parser.parse_args()


    # With transform 
    train_transform = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomRotation90(),
    MinMaxNormalization(),
    ToTensor(),
    ])

    # For validation or testing, a simpler pipeline (only normalization and conversion to tensor) can be used:
    val_transform = Compose([
        MinMaxNormalization(),
        ToTensor()
    ])


    train(args, train_transform=train_transform,
          valid_transform=val_transform)  # Call the train function

    # train(args)