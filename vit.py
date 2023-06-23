#!/usr/bin/env python
# coding: utf-8

import torch
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch import optim
import matplotlib.pyplot as plt

import timm
import mlflow
import os

from torch import nn
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

import train 
import datetime as dt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CreateImageDataset(Dataset):
    def __init__(self, labels, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_labels = labels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx]['grade_encode']
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
trainpath = 'data/Training/'
valpath = 'data/Validation/'
train_imagepath = os.path.join(trainpath, 'images')
val_imagepath = os.path.join(valpath,'images')

train_label_set = pd.read_csv("train_small.csv")
val_label_set = pd.read_csv("val_small.csv")

def grade_encoding(x):
    #if x == '1++':
    #    return 0
    # elif x == '1+':
    #     return 1
    if x == 1:
        return 0
    elif x == 2:
        return 1
    elif x== 3:
        return 2
    return 0

train_label_set['grade_encode'] = train_label_set['grade'].apply(grade_encoding)
val_label_set['grade_encode'] = val_label_set['grade'].apply(grade_encoding)


#Define hyperparameters
batch_size = 16
lr = 0.01
epochs = 20
log_epoch = 5
num_workers = 4
num_classes = 3
pretrained = True
model_name = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'

transformation = transforms.Compose([
transforms.Resize([224,224]),
transforms.ToTensor(),
])

model = timm.create_model(model_name, pretrained=pretrained, num_classes = num_classes)
model = model.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lr)
scheduler = ReduceLROnPlateau(optimizer, patience = 4, factor = 0.1, threshold = 0.001)
train_dataset = CreateImageDataset(train_label_set, train_imagepath, transform=transformation)
valid_dataset = CreateImageDataset(val_label_set, val_imagepath, transform=transformation)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

params_train = {
'num_epochs':epochs,
'optimizer':optimizer,
'loss_func':loss_func,
'train_dl':train_dataloader,
'val_dl':valid_dataloader,
'sanity_check':False,
'lr_scheduler':scheduler,
'log_epoch':log_epoch,
}

params_model = {
    'batch size':batch_size,
    'lr':lr,
    'model name':model_name,
    'pretrained':pretrained,
}

experiment_name = "ViT"
mlflow.set_experiment(experiment_name)
run_name = "ViT" + str(dt.today())

with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_params(params_train)
    mlflow.log_params(params_model)
    model, train_acc, val_acc, train_loss, val_loss = train.training(model, params_train)
    mlflow.pytorch.log_model(model, "Final")

    #plot the curves
    plt.plot(train_acc, label = 'train_acc')
    plt.plot(val_acc, label = 'val_acc')
    plt.plot(train_loss, label = 'train_loss')
    plt.plot(val_loss, label = 'val_loss')
    plt.legend()
    plt.title('Accuracy and Loss Plots')
    plt.show()