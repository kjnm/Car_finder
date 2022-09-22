import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from dataset_cars_classifier import CarsImageDataset, CarsImageDataset_for_test
import torchvision.models as models
from torch.utils.data import random_split, Subset
from effnetv2 import effnetv2_xs
from PIL import Image
import shutil
import torch.optim as optim
import os
from tqdm import tqdm

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def filter_dataset(dataset):
    good_idxs = []
    print("Filtering dataset:")
    for idx, path in tqdm(enumerate(dataset.data)):
        im = Image.open(path)
        width, height = im.size

        prop = width / height

        if 1.3 < prop < 1.8:
            good_idxs.append(idx)

    return Subset(dataset, good_idxs)

def prepare_data(flip = False, test_part = 0.1):
    wholeset = CarsImageDataset()

    wholeset = filter_dataset(wholeset)

    test_len = int(len(wholeset) * test_part)
    train_len = len(wholeset) - test_len

    testset, trainset = random_split(wholeset, [test_len, train_len], generator=torch.Generator().manual_seed(4))

    if flip:
        return trainset, testset
    else:
        return testset, trainset


def train(dev, trainset):

    t_max = 100
    batch_size = 32

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)

    net = models.resnet34(num_classes =2)

    net.to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.NAdam(net.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    net.train()

    for epoch in range(t_max):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(dev)
            labels = labels.to(dev)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            printing_frequency = 100
            if i % printing_frequency == printing_frequency - 1:
                print('[%d, %5d] loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / printing_frequency))
                running_loss = 0.0
        scheduler.step()

    print('Finished Training')
    PATH = 'model.pth'
    torch.save(net.state_dict(), PATH)

def test(dev,testset, PATH, remove = True):
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=5)

    net = models.resnet34(num_classes =2)
    #net = effnetv2_xs(num_classes =2)
    net.to(dev)
    net.load_state_dict(torch.load(PATH))
    net.eval()
    dirpath = "D:\imagesType_check"

    if remove:
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)

        os.mkdir(r'D:\imagesType_check')
        os.mkdir(r'D:\imagesType_check\part')
        os.mkdir(r'D:\imagesType_check\full')

    testloader.dataset.dataset.dataset.apply_transform = False
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data

            images = images.to(dev)
            labels = labels.to(dev)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if predicted[0] != labels[0]:

                path = testloader.dataset.dataset.dataset.data[testloader.dataset.dataset.indices[testloader.dataset.indices[i]]]
                print(path, outputs.data)

                shutil.copy(path, "D:\imagesType_check"+path[13:])


    print('Accuracy of the network on the test images: %f %%' % (
        100 * correct / total))


def predict(dev, PATH):

    testset = CarsImageDataset_for_test(r'D:\images_All_Full')
    testset = filter_dataset(testset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=5)
    net = effnetv2_xs(num_classes =2)
    net.to(dev)
    net.load_state_dict(torch.load(PATH))
    net.eval()
    dirpath = "D:\imagesType_predict_All_Full"

    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)

    os.mkdir(r'D:\imagesType_predict_All_Full')
    os.mkdir(r'D:\imagesType_predict_All_Full\part')
    os.mkdir(r'D:\imagesType_predict_All_Full\full')

    testloader.dataset.dataset.apply_transform = False
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in tqdm((enumerate(testloader, 0))):

            data = data.to(dev)
            # calculate outputs by running images through the network
            outputs = net(data)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            path = testloader.dataset.dataset.data[testloader.dataset.indices[i]]
            if predicted[0]:
                shutil.copy(path, dirpath + r'\\full\\' + os.path.basename(path))
            else:
                shutil.copy(path, dirpath + r'\\part\\' + os.path.basename(path))


def tunning():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    testset, trainset = prepare_data(test_part=0.5)
    train(dev, trainset)
    test(dev, testset, "model.pth")

    train(dev, testset)
    test(dev, trainset, "model.pth", remove=False)

def main():

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    testset, trainset = prepare_data(test_part = .1)
    train(dev, trainset)
    test(dev, testset, "./model.pth")
    #predict(dev, "eff_v2_final.pth")

if __name__ == '__main__':
    #tunning()
    main()




