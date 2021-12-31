import os
import numpy as np
from torch.utils.data import Dataset
from os.path import join
import torchvision.transforms as transforms
import cv2
import torch.nn.functional as F
import random
import albumentations as A

IMAGE_SIZE = (400, 300)

transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=10, border_mode= cv2.BORDER_REPLICATE, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
    A.OpticalDistortion(p=0.5, distort_limit=(-0.3, 0.3), shift_limit=(-0.02, 0.02), border_mode= cv2.BORDER_REPLICATE),
    A.CoarseDropout(p=0.3, max_holes=50, max_height=3, max_width=3)
])

class CarsImageDataset(Dataset):
    def __init__(self, img_dir = "D:\imagesType"):
        self.img_dir = img_dir

        self.data = []
        self.target = []
        self.apply_transform = True

        for file in os.listdir(img_dir + r'\part'):
            self.data.append(join(img_dir + r'\part', file))
            self.target.append(0)

        for file in os.listdir(img_dir + r'\full'):
            self.data.append(join(img_dir + r'\full', file))
            self.target.append(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = cv2.imread(img_path)
        image = np.array(image)
        image = cv2.resize(image, dsize=(IMAGE_SIZE[0], int(image.shape[0]/image.shape[1] * IMAGE_SIZE[0])))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



        if self.apply_transform:
            image = transform(image=image)["image"]

        t = transforms.Compose([transforms.ToTensor()])
        image = t(image)

        max_width = IMAGE_SIZE[0]
        max_height = IMAGE_SIZE[1]

        if max_width - image.size(2) > 0:
            dx = random.randrange(max_width - image.size(2))
        else:
            dx = 0
        if max_height - image.size(1) > 0:
            dy = random.randrange(max_height - image.size(1))
        else:
            dy = 0

        image = F.pad(image, [dx, max_width - image.size(2) - dx, dy, max_height - image.size(1) - dy], mode="replicate")

        label = self.target[idx]

        return image, label


class CarsImageDataset_for_test(Dataset):
    def __init__(self, img_dir = r'D:\images\full'):
        self.img_dir = img_dir
        self.data = []
        self.apply_transform = True


        for file in os.listdir(img_dir):
            self.data.append(join(img_dir, file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = cv2.imread(img_path)
        image = np.array(image)
        image = cv2.resize(image, dsize=(IMAGE_SIZE[0], int(image.shape[0]/image.shape[1] * IMAGE_SIZE[0])))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



        if self.apply_transform:
            image = transform(image=image)["image"]

        t = transforms.Compose([transforms.ToTensor()])
        image = t(image)

        max_width = IMAGE_SIZE[0]
        max_height = IMAGE_SIZE[1]

        if max_width - image.size(2) > 0:
            dx = random.randrange(max_width - image.size(2))
        else:
            dx = 0
        if max_height - image.size(1) > 0:
            dy = random.randrange(max_height - image.size(1))
        else:
            dy = 0

        image = F.pad(image, [dx, max_width - image.size(2) - dx, dy, max_height - image.size(1) - dy], mode="replicate")

        return image