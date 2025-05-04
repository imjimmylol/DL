from src.model import UNet
from file.evaluator import evaluation_model
from src.dataloader import ClevrDataset
from torchvision import transforms

DATADIR = "./iclevr"
INDEXFILE =  "./file/test.json"
EVALMODEL = evaluation_model


MODELDIR = "./checkpoints/ckot_epoch_5.pth"





