import json
import os

from main_car_model import train, test, convert_to_mobile, model_benchmark
from car_category_dataset import get_Cars_Category_Dataset_Division
import torch
import torchvision.models as models
import  effnetv2
def main():

    run_data = [
        {"dir_name": "moblie_v3_l_60epoch_batch64_0.5", "epoch": 60, "model": models.mobilenet_v3_large, "resolution" : (400,300), "batch_size" : 64, "loss_frequancy" : 1, "used_data" : 0.5},
        {"dir_name": "moblie_v3_l_30epoch_batch64_1.0", "epoch": 30, "model": models.mobilenet_v3_large, "resolution" : (400,300), "batch_size" : 64, "loss_frequancy" : 1, "used_data" : 1.0},
        #{"dir_name": "moblie_v3_l_60epoch", "epoch": 60, "model": models.mobilenet_v3_large(num_classes =29)},
        #{"dir_name": "moblie_v3_l_10epoch", "epoch": 10, "model": models.mobilenet_v3_large(num_classes =29)},
        #{"dir_name": "moblie_v3_l_20epoch", "epoch": 20, "model": models.mobilenet_v3_large(num_classes =29)},
    ]

    only_test = False


    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"


    for model in run_data:
        test_dataset, train_dataset = get_Cars_Category_Dataset_Division(
            image_size=(model["resolution"] if model["resolution"] else (400,300)),
            path = r'C:\Users\krzys\Documents\scrapy-otomoto\otomoto.json',
            imagePath = r'D:\imagesType_predict_All_Full\full',
            used_part=model["used_data"]
            #path = r'C:\Users\krzys\Documents\scrapy-otomoto\otomoto2.json',
            #imagePath = r'D:\imagesType_predictAudi\full'
        )

        path = "experiments/" + model["dir_name"]
        if not only_test:
            os.mkdir(path)
            losses = train(dev, train_dataset, PATH=path +'/model.pth', t_max=model["epoch"], net=model["model"](num_classes = len(train_dataset.category_name)), batch_size=model["batch_size"], loss_frequancy=model["loss_frequancy"])
        res = test(dev, test_dataset, path +'/model.pth', net=model["model"](num_classes = len(train_dataset.category_name)))
        res["details"] = model_benchmark(dev, path + '/model.pth', net=model["model"](num_classes = len(train_dataset.category_name)))
        convert_to_mobile(path +'/model.pth', dir=path, net=model["model"](num_classes = len(train_dataset.category_name)))

        if not only_test:
            res["losses"] = losses

        model["model"] = type(model['model']).__name__
        res["model"] = model

        with open(path +'/result.json', 'w') as f:
            json.dump(res, f)

if __name__ == '__main__':
    main()
