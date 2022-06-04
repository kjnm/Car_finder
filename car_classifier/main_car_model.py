import os
import shutil
import time
from os.path import join

from torch.cuda.amp import GradScaler, autocast

from car_classifier.car_category_dataset import get_Cars_Category_Dataset_Division, Cars_Category_Dataset

import numpy as np
import torch
from effnetv2 import effnetv2_xs
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision.models as models
import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
from line_profiler_pycharm import profile
import wandb

@profile
def train(dev, trainset, testset, PATH = 'model.pth', t_max = 10, net= models.mobilenet_v3_large(num_classes =29), batch_size=32, loss_frequancy = 1 ):

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)

    net.to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.NAdam(net.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    epoch_losses = []
    scaler = GradScaler()
    for epoch in tqdm(range(t_max)):

        running_loss = 0.0
        net.train()
        losses = []
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(dev)
            labels = labels.to(dev)
            optimizer.zero_grad()
            with autocast(enabled = True):
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            if (i+1) % loss_frequancy == 0:
                scaler.step(optimizer)
                scaler.update()
            running_loss += loss.item()

            printing_frequency = 100
            if i % printing_frequency == printing_frequency - 1:
                print('[%d, %5d] loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / printing_frequency))
                losses.append(running_loss / printing_frequency)
                running_loss = 0.0


        wandb.log({"loss" : sum(losses)/len(losses), "epoch": epoch +1})

        epoch_losses.append(sum(losses)/len(losses))
        torch.save(net.state_dict(), PATH)
        quick_test(dev, testset, net, batch_size * 2, epoch+1)
        quick_test(dev, testset, net, 1, epoch+1)

        scheduler.step()

    print('Finished Training')
    torch.save(net.state_dict(), PATH)
    return epoch_losses


def test(dev,testset, PATH, net = models.mobilenet_v3_large(num_classes =29)):
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=5)

    net.to(dev)
    net.load_state_dict(torch.load(PATH))
    net.eval()

    testset.apply_transform = False
    correct = 0

    correct_class = {}
    category_correct = {}
    category_total = {}
    revert_encode = {}
    errors_filenames = []

    for idx,category in enumerate(testset.category_name):
        category_correct[category] = 0.
        category_total[category] = 0.
        revert_encode[idx] = category

    top_class = list(range(1,6))
    for i in top_class:
        correct_class[i] = 0

    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader, 0), total=len(testset)):
            images, labels = data

            images = images.to(dev)
            labels = labels.to(dev)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            for k in top_class:
                correct_class[k]+= labels[0] in torch.topk(outputs.data, k).indices

            category_total[revert_encode[labels[0].item()]] += 1
            if predicted == labels:
                category_correct[revert_encode[labels[0].item()]] += 1
            else:
                errors_filenames.append(testset.data[i])

            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print('Accuracy of the network on the test images: %f %%' % (
        100 * correct / total))


    for k in top_class:
        print('Accuracy of top@'+ str(k) +' the network on the test images: %f %%' % (100 * correct_class[k] / total))

    for i in top_class:
        correct_class[i] /= total


    for category in testset.category_name:
        category_correct[category] /= category_total[category]

    correct_class["By Category"] = category_correct
    correct_class["Errors Filenames"] = errors_filenames

    return correct_class

def quick_test(dev, testset, net, batch_size, epoch):
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=5)

    net.eval()

    testset.apply_transform = False
    correct = 0

    correct_class = {}
    category_correct = {}
    category_total = {}
    revert_encode = {}
    errors_filenames = []

    for idx,category in enumerate(testset.category_name):
        category_correct[category] = 0.
        category_total[category] = 0.
        revert_encode[idx] = category

    top_class = list(range(1,6))
    for i in top_class:
        correct_class[i] = 0

    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader, 0), total=len(testset)//batch_size):
            images, labels = data

            images = images.to(dev)
            labels = labels.to(dev)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            for nr_batch in range(min(batch_size,labels.size(0))):

                for k in top_class:
                    correct_class[k]+= labels[nr_batch] in torch.topk(outputs.data[nr_batch], k).indices

                category_total[revert_encode[labels[nr_batch].item()]] += 1
                if predicted[nr_batch] == labels[nr_batch]:
                    category_correct[revert_encode[labels[nr_batch].item()]] += 1
                else:
                    errors_filenames.append(testset.data[i])

            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print('Accuracy of the network on the test images: %f %%' % (
        100 * correct / total))


    for k in top_class:
        print('Accuracy of top@'+ str(k) +' the network on the test images: %f %%' % (100 * correct_class[k] / total))
        wandb.log({"acc@"+ str(k) : (100 * correct_class[k] / total), "epoch": epoch})


    for i in top_class:
        correct_class[i] /= total

    for category in testset.category_name:
        category_correct[category] /= category_total[category]

    for q in np.linspace(0, 1, 11):
        np.quantile(list(category_correct.values()), q=q)
        wandb.log({"class@" + "{:.1f}".format(q): (100 * np.quantile(list(category_correct.values()), q=q)), "epoch": epoch})

    correct_class["By Category"] = category_correct
    correct_class["Errors Filenames"] = errors_filenames

    return correct_class

def convert_to_mobile(PATH, dir, net):

    net.to("cpu")
    net.load_state_dict(torch.load(PATH))
    net.eval()
    example = torch.rand(1, 3, 400, 300)
    traced_script_module = torch.jit.trace(net, example)
    optimized_traced_model = optimize_for_mobile(traced_script_module)
    optimized_traced_model._save_for_lite_interpreter(dir + "/moblie_model.ptl")


def model_benchmark(dev, PATH, net):

    size = 100
    examples = [torch.rand(1, 3, 400, 300) for _ in range(size)]
    net.to("cpu")
    net.load_state_dict(torch.load(PATH))
    net.eval()

    res = {}
    res["Number of parameters"] = {}
    res["Interferance time"] = {}
    start = time.time()
    for i in range(size):
        net(examples[i])

    end = time.time()
    cpuTime = end -start

    res["Interferance time"]["cpu time(ms)"] = cpuTime/size * 1000

    net.train()

    res["Number of parameters"]["trainable"] = sum(p.numel() for p in net.parameters() if p.requires_grad)
    res["Number of parameters"]["total"] = sum(p.numel() for p in net.parameters())

    net.eval()

    if dev != "cpu":
        net.to("cuda:0")

        for i in range(size):
            examples[i] = examples[i].to("cuda:0")

        start = time.time()
        for i in range(size):
            net(examples[i])

        end = time.time()
        gpuTime = end -start

        res["Interferance time"]["gpu time(ms)"] = gpuTime/size * 1000

    return res

def predict(dev, PATH, net, imgsPath):
    test_dataset, train_dataset = get_Cars_Category_Dataset_Division(
        image_size= (400, 300),
        path=r'C:\Users\krzys\Documents\scrapy-otomoto\otomoto.json',
        imagePath=r'D:\imagesType_predict_All_Full\full'
        # path = r'C:\Users\krzys\Documents\scrapy-otomoto\otomoto2.json',
        # imagePath = r'D:\imagesType_predictAudi\full'
    )

    datap = []
    for file in os.listdir(imgsPath):
        datap.append((join(imgsPath, file), 0))

    ds = Cars_Category_Dataset(data=datap,category_name=[])
    top_class = list(range(1,6))

    ds.apply_transform = False
    net.to(dev)
    net.load_state_dict(torch.load(PATH))
    net.eval()

    with torch.no_grad():
        for i in range(len(ds)):
            images, labels = ds[i]

            images = images.to(dev)
            # calculate outputs by running images through the network
            outputs = net(torch.unsqueeze(images,0))
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            print(ds.data[i])

            for x in torch.topk(outputs.data, 5).indices[0]:
                print(test_dataset.category_name[x])


def main():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    #convert_to_mobile(dev, "model.pth")

    test_dataset, train_dataset = get_Cars_Category_Dataset_Division()

    train(dev, train_dataset, 'model_moblie_v3_l_10epoch.pth')
    test(dev, test_dataset, 'model_moblie_v3_l_10epoch.pth')


if __name__ == '__main__':
    predict("cuda:0", "experiments/resnet34_l_30epoch_batch32ALL/model.pth", models.resnet34(num_classes = 366), "D:\images_no_label")

