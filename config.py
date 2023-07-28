import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


SAVED_MODEL_PATH = './models/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 10000
LEARNING_RATE = 0.001
BATCH_SIZE = 2
ROOT_DIR_TRAIN = './data/train'
ROOT_DIR_VAL = './data/val'
ROOT_DIR_TEST = './data/test'
LOAD_MODEL = False
PATH_TO_MODEL = './model.pth'