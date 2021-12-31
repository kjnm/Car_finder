from torch.utils.data import Dataset
import json
import os
import numpy as np
from torch.utils.data import Dataset
from os.path import join
import torchvision.transforms as transforms
import cv2
import torch.nn.functional as F
import random
import albumentations as A


transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=10, border_mode= cv2.BORDER_REPLICATE, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
    A.OpticalDistortion(p=0.5, distort_limit=(-0.3, 0.3), shift_limit=(-0.02, 0.02), border_mode= cv2.BORDER_REPLICATE),
    A.CoarseDropout(p=0.3, max_holes=50, max_height=3, max_width=3)
])

def get_Cars_Category_Dataset_Division(
        test_part = 0.1,
        used_part = 1.0,
        path = r'C:\Users\krzys\Documents\scrapy-otomoto\otomoto2.json',
        imagePath = r'D:\imagesType_predictAudi\full',
        image_size = (400, 300)
):


    with open(path, 'r') as f:
        data = json.load(f)

    images = {}
    car = {}

    for item in data:
        if len(item) == 2:
            if len(item["images"]) == 0:
                #print(item['image_urls'][0])
                continue

            images[item["images"][0]["url"][:]] = item["images"][0]["path"]
        else:

            if "brand" in item and "model" in item:

                category_name = item["brand"] + " " + item["model"]

                if "generation" in item:
                    category_name  = category_name + " " + item["generation"]

                if category_name in car:
                    car[category_name].append(item)
                else:
                    car[category_name] = [item]

    del data

    test_data = []
    train_data = []

    car = {k: v for k, v in car.items() if len(v) >= 100}

    for idx, category in enumerate(car):
        for i, an in enumerate(car[category]):
            if 'images2s' in an:
                for img in an['images2s']:
                    if img in images and os.path.exists(imagePath + "/" + images[img]):

                        data = (imagePath + "/" + images[img], idx)

                        if i <= test_part * len(car[category]):
                            test_data.append(data)
                        else:
                            if i < test_part * len(car[category]) + used_part * (1-test_part) * len(car[category]):
                                train_data.append(data)

    category_list = [x for x in car.keys()]

    print("Loaded datasets")
    print("Len of test=", len(test_data), "Len of train=", len(train_data))
    return Cars_Category_Dataset(test_data, category_list, image_size), Cars_Category_Dataset(train_data, category_list, image_size)

class Cars_Category_Dataset(Dataset):
    def __init__(self, data, category_name, image_size = (400, 300)):
        self.data = data
        self.apply_transform = True
        self.category_name = category_name
        self.IMAGE_SIZE = image_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx][0]
        image = cv2.imread(img_path)
        image = np.array(image)
        image = cv2.resize(image, dsize=(self.IMAGE_SIZE[0], int(image.shape[0]/image.shape[1] * self.IMAGE_SIZE[0])))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.apply_transform:
            image = transform(image=image)["image"]

        t = transforms.Compose([transforms.ToTensor()])
        image = t(image)

        max_width = self.IMAGE_SIZE[0]
        max_height = self.IMAGE_SIZE[1]

        if max_width - image.size(2) > 0:
            if self.apply_transform:
                dx = random.randrange(max_width - image.size(2))
            else:
                dx = (max_width - image.size(2))//2
        else:
            dx = 0
        if max_height - image.size(1) > 0:
            if self.apply_transform:
                dy = random.randrange(max_height - image.size(1))
            else:
                dy = (max_height - image.size(1)) // 2
        else:
            dy = 0

        image = F.pad(image, [dx, max_width - image.size(2) - dx, dy, max_height - image.size(1) - dy], mode="replicate")

        return image, self.data[idx][1]