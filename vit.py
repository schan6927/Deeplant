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
import time

import train 
from datetime import datetime as dt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def grade_encoding(x):
    if x == '1++':
        return 0
    elif x == '1+':
         return 1
    elif x == '1':
        return 2
    elif x == '2':
        return 3
    elif x == '3':
        return 4
    return 0

class CreateImageDataset(Dataset):
    def __init__(self, labels, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_labels = labels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        
        label = self.img_labels.iloc[idx]['grade_encode']
        grade = self.img_labels.iloc[idx]['grade']
        img_folder = f'grade_{grade}'
        img_path = os.path.join(self.img_dir, img_folder)
        img_path = os.path.join(img_path, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

#Define data pathes
imageSize = 448
homepath = 'data'
datapath = os.path.join(homepath,str(imageSize))
trainpath = os.path.join(datapath,'Training')
valpath = os.path.join(datapath,'Valid')

train_label_set = pd.read_csv(f'{datapath}/train.csv')
val_label_set = pd.read_csv(f'{datapath}/valid.csv')

train_label_set['grade_encode'] = train_label_set['grade'].apply(grade_encoding)
val_label_set['grade_encode'] = val_label_set['grade'].apply(grade_encoding)

#Define hyperparameters
batch_size = 16
lr = 0.0001
epochs = 10

#Define kfold
fold = 5

log_epoch = 5
num_workers = 4
num_classes = 5
pretrained = True
model_name = 'vit_base_patch32_clip_448.laion2b_ft_in12k_in1k'

load_run = False
logged_model = 'runs:/523f68657d884879844be1c409bd96c0/best'

if load_run == True:
    model = mlflow.pytorch.load_model(logged_model)
else:
    model = timm.create_model(model_name, pretrained=pretrained, num_classes = num_classes)

#Define input transform
transformation = transforms.Compose([
transforms.RandomHorizontalFlip(p=0.3),
transforms.RandomVerticalFlip(p=0.3),
transforms.RandomRotation((-20,20)),
transforms.ToTensor(),
])

model = model.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lr)
scheduler = ReduceLROnPlateau(optimizer, patience = 4, factor = 0.1, threshold = 0.001)
train_dataset = CreateImageDataset(train_label_set, trainpath, transform=transformation)
valid_dataset = CreateImageDataset(val_label_set, valpath, transform=transformation)
#train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

params_train = {
'num_epochs':epochs,
'optimizer':optimizer,
'loss_func':loss_func,
'sanity_check':False,
'lr_scheduler':scheduler,
'log_epoch':log_epoch,
'fold':fold,
'train_dataset':train_dataset,
'valid_dataset':valid_dataset,
'batch_size':batch_size,
}

experiment_name = "ViT"
mlflow.set_experiment(experiment_name)

now = dt.now()
date_time_string = now.strftime("%Y-%m-%d %H:%M:%S")
run_name = "ViT" + str(date_time_string)

with mlflow.start_run(run_name=run_name) as run:
    print(run.info.run_id)
    if load_run == False:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param('pretrained', pretrained)
    else:
        mlflow.log_param("model_name", logged_model)
        mlflow.log_param("pretrained", True)

    mlflow.log_param("num_epochs", epochs)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param('batch_size', batch_size)
    mlflow.log_param("image_size", imageSize)

    mlflow.log_param("optimizer", optimizer)
    mlflow.log_param("loss_func", loss_func)
    mlflow.log_param("lr_scheduler", scheduler)

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


