import json
import os

from torch import optim

from main_car_model import train, test, convert_to_mobile, model_benchmark
from car_category_dataset import get_Cars_Category_Dataset_Division
import torch
import torchvision.models as models
import effnetv2
import timm
import wandb
from moblienetv1 import Moblienetv1
def main():

    run_data = [
        #{"dir_name": "test1o", "epoch": 1, "model":  models.mobilenet_v3_large, "resolution" : (400,300), "batch_size" : 64, "loss_frequancy" : 1, "used_data" : 0.01, "pretrained": False},
        #{"dir_name": "test2o", "epoch": 1, "model":  models.mobilenet_v3_large, "resolution" : (400,300), "batch_size" : 64, "loss_frequancy" : 1, "used_data" : 0.01, "pretrained": False},
        #{"dir_name": "resnet18_30epoch_Adam", "epoch": 30, "model":  models.resnet18, "resolution": (400, 300),   "batch_size": 32, "loss_frequancy": 2, "used_data": 1.0, "optimizer": optim.AdamW},
        #{"dir_name": "resnet18_30epochAdam_64batch", "epoch": 30, "model":  models.resnet18, "resolution" : (400, 300), "batch_size" : 64, "loss_frequancy" : 1, "used_data" : 1.0, "optimizer" : optim.AdamW  },
        #{"dir_name": "resnet34_30epochAdam", "epoch": 30, "model":  models.resnet34, "resolution" : (400, 300), "batch_size" : 32, "loss_frequancy" : 2, "used_data" : 1.0, "optimizer" : optim.AdamW  },
        #{"dir_name": "mobilenet_v3_small_30epochAdam", "epoch": 30, "model": models.mobilenet_v3_small, "resolution": (400, 300),   "batch_size": 64, "loss_frequancy": 1, "used_data": 1.0, "optimizer": optim.AdamW},
        #{"dir_name": "mobilenet_v2_30epochAdam", "epoch": 30, "model": models.mobilenet_v2, "resolution": (400, 300), "batch_size": 32, "loss_frequancy": 2, "used_data": 1.0, "optimizer": optim.AdamW},

        #{"dir_name": "lcnet_150_30epochAdam", "epoch": 30, "model": "lcnet_150", "resolution": (400, 300), "batch_size": 64, "loss_frequancy": 1, "used_data": 1., "optimizer": optim.AdamW},
        #{"dir_name": "lcnet_100_30epochAdam", "epoch": 30, "model": "lcnet_100", "resolution": (400, 300), "batch_size": 64, "loss_frequancy": 1, "used_data": 1., "optimizer": optim.AdamW},
        #{"dir_name": "lcnet_75_30epochAdam", "epoch": 30, "model": "lcnet_075", "resolution": (400, 300), "batch_size": 64, "loss_frequancy": 1, "used_data": 1., "optimizer": optim.AdamW},
        #{"dir_name": "lcnet_50_30epochAdam", "epoch": 30, "model": "lcnet_050", "resolution": (400, 300), "batch_size": 64, "loss_frequancy": 1, "used_data": 1., "optimizer": optim.AdamW},
        #{"dir_name": "lcnet_035_30epochAdam", "epoch": 30, "model": "lcnet_035", "resolution": (400, 300), "batch_size": 64, "loss_frequancy": 1, "used_data": 1., "optimizer": optim.AdamW},
        #{"dir_name": "fbnetv3_d_30epochAdam", "epoch": 30, "model": "fbnetv3_d", "resolution": (400, 300), "batch_size": 32, "loss_frequancy": 2, "used_data": 1., "optimizer": optim.AdamW},
        #{"dir_name": "fbnetv3_b_30epochAdam", "epoch": 30, "model": "fbnetv3_b", "resolution": (400, 300), "batch_size": 32, "loss_frequancy": 2, "used_data": 1., "optimizer": optim.AdamW},
        #{"dir_name": "mobilenetv3_large_075_30epochAdam", "epoch": 30, "model": "mobilenetv3_large_075", "resolution": (400, 300), "batch_size": 32, "loss_frequancy": 2, "used_data": 1., "optimizer": optim.AdamW},
        #{"dir_name": "mobilenetv3_large_100_30epochAdam", "epoch": 30, "model": "mobilenetv3_large_100", "resolution": (400, 300), "batch_size": 32, "loss_frequancy": 2, "used_data": 1., "optimizer": optim.AdamW},
        #{"dir_name": "mobilenetv3_large_100_miil_30epochAdam", "epoch": 30, "model": "mobilenetv3_large_100_miil", "resolution": (400, 300), "batch_size": 32, "loss_frequancy": 2, "used_data": 1., "optimizer": optim.AdamW},
        #{"dir_name": "mobilenetv3_small_050_30epochAdam", "epoch": 30, "model": "mobilenetv3_small_050", "resolution": (400, 300), "batch_size": 64, "loss_frequancy": 1, "used_data": 1., "optimizer": optim.AdamW},
        #{"dir_name": "mobilenetv3_small_075_30epochAdam", "epoch": 30, "model": "mobilenetv3_small_075", "resolution": (400, 300), "batch_size": 64, "loss_frequancy": 1, "used_data": 1., "optimizer": optim.AdamW},
        #{"dir_name": "mobilenetv3_small_100_30epochAdam", "epoch": 30, "model": "mobilenetv3_small_100", "resolution": (400, 300), "batch_size": 64, "loss_frequancy": 1, "used_data": 1., "optimizer": optim.AdamW},
        #{"dir_name": "TEST", "epoch": 5, "model": "mobilenetv3_large_100", "resolution": (400, 300), "batch_size": 32, "loss_frequancy": 2, "used_data": 1., "optimizer": optim.AdamW},
        {"dir_name": "mobilenetv3_large_100_10epochSGD+mom+wd", "epoch": 10, "model": "mobilenetv3_large_100", "resolution": (400, 300), "batch_size": 32, "loss_frequancy": 2, "used_data": 1., "optimizer": optim.SGD},
        #{"dir_name": "mobilenetv3_large_100_50epochAdam", "epoch": 50, "model": "mobilenetv3_large_100", "resolution": (400, 300), "batch_size": 32, "loss_frequancy": 2, "used_data": 1., "optimizer": optim.AdamW},

        #{"dir_name": "fbnetv3_g_30epochAdam", "epoch": 30, "model": "fbnetv3_g", "resolution": (400, 300), "batch_size": 16, "loss_frequancy": 4, "used_data": 1., "optimizer": optim.AdamW},

    ]

    only_test = False
    enable_wandb = True

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"


    for model in run_data:

        with wandb.init(project="Car finder", entity="kjnm", name=model["dir_name"], mode ="online" if enable_wandb else "disabled"):
            wandb.config.update(model)

            test_dataset, train_dataset = get_Cars_Category_Dataset_Division(
                image_size=(model["resolution"] if model["resolution"] else (400,300)),
                path = r'C:\Users\krzys\Documents\scrapy-otomoto\otomoto.json',
                imagePath = r'D:\imagesType_predict_All_Full\full',
                used_part=model["used_data"],
            )

            if not isinstance(model["model"], str):
                net = model["model"](num_classes = len(train_dataset.category_name)) #torch vision
            else:
                net = timm.create_model(model["model"], num_classes=len(train_dataset.category_name)) #timm

            #net = Moblienetv1(num_classes = len(train_dataset.category_name))

            wandb.watch(net, log='all')

            path = "experiments/" + model["dir_name"]
            if not only_test:
                os.mkdir(path)
                losses = train(dev, train_dataset, test_dataset, PATH=path +'/model.pth', t_max=model["epoch"], net=net, batch_size=model["batch_size"], loss_frequancy=model["loss_frequancy"], optimizer_f=model["optimizer"])
            res = test(dev, test_dataset, path +'/model.pth', net=net)
            res["details"] = model_benchmark(dev, path + '/model.pth', net=net)
            convert_to_mobile(path +'/model.pth', dir=path, net=net)

            if not only_test:
                res["losses"] = losses
            if not isinstance(model["model"], str):
                model["model"] = type(model['model']).__name__

            model["optimizer"] = type(model['optimizer']).__name__
            res["model"] = model

            with open(path +'/result.json', 'w') as f:
                json.dump(res, f)

if __name__ == '__main__':
    main()
