import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


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
TRANSFORM_TRAIN = A.Compose([A.Resize(1088, 608),
                             A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                             A.HorizontalFlip(p=0.5),
                             A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                             A.RandomBrightnessContrast(p=0.5),
                             A.GaussianBlur(p=0.5),
                             A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                             A.OpticalDistortion(p=0.5),
                             A.MotionBlur(blur_limit=15, p=0.5),
                             ToTensorV2()])
TRANSFORM_VAL_TEST = A.Compose([A.Resize(1088, 608),
                                A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                ToTensorV2()])
