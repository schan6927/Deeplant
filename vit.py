#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
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
    

# calculate the metric per mini-batch
def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects

# calculate the loss per mini-batch
def loss_batch(loss_func, output, target, opt=None):

    loss_b = loss_func(output, target)
    metric_b = metric_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()

    return loss_b.item(), metric_b

# calculate the loss per epochs
def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metrics = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
        running_loss += loss_b

        if metric_b is not None:
            running_metrics += metric_b

        if sanity_check is True:
            break

    loss = running_loss / len_data
    metric = running_metrics / len_data
    return loss, metric


def training(model, params):
    num_epochs=params['num_epochs']
    loss_func=params['loss_func']
    optimizer=params['optimizer']
    train_dl=params['train_dl']
    val_dl=params['val_dl']
    scheduler=params['lr_scheduler']
    path2weights=params['path2weights']
    train_loss, val_loss, train_metric, val_metric =[], [], [], []
    best_acc = 0

    for epoch in tqdm(range(num_epochs)):
        #training
        model.train()
        loss, metric = loss_epoch(model, loss_func, train_dl, False, optimizer)
        mlflow.log_metric("train loss", loss)
        mlflow.log_metric("train accuracy", metric)
        train_loss.append(loss)
        train_metric.append(metric)

        #validation
        model.eval()
        with torch.no_grad():
            loss, metric = loss_epoch(model, loss_func, val_dl, False, optimizer)
            mlflow.log_metric("val loss", loss)
            mlflow.log_metric("val accuracy", metric)
            val_loss.append(loss)
            val_metric.append(metric)
        scheduler.step(val_loss[-1])

        #saving best model
        if val_metric[-1]>best_acc:
            best_acc = val_metric[-1]
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':val_loss[-1],
                'acc':val_metric[-1]
            }, path2weights)
        print('The Validation Loss is {} and the validation accuracy is {}'.format(val_loss[-1],val_metric[-1]))
        print('The Training Loss is {} and the training accuracy is {}'.format(train_loss[-1],train_metric[-1]))

    return model, train_metric, val_metric, train_loss, val_loss




trainpath = '../Training/'
valpath = '../Validation/'
train_imagepath = os.path.join(trainpath, 'images')
val_imagepath = os.path.join(valpath,'images')

train_labelpath = os.path.join(trainpath, 'labels')
val_labelpath = os.path.join(valpath, 'labels')

train_label_set = pd.read_csv("train.csv")
val_label_set = pd.read_csv("valid.csv")

def grade_encoding(x):
    if x == '1++':
        return 0
    elif x == '1+':
        return 1
    elif x == '1':
        return 2
    elif x == '2':
        return 3
    elif x== '3':
        return 4
    return 0

train_label_set['grade_encode'] = train_label_set['grade'].apply(grade_encoding)
val_label_set['grade_encode'] = val_label_set['grade'].apply(grade_encoding)


#Define hyperparameters
batch_size = 32
lr = 0.01
epochs = 100
num_workers = 8
num_classes = 5
pretrained = True
model_name = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'

transformation = transforms.Compose([
transforms.Resize([224,224]),
transforms.ToTensor(),
])

model = timm.create_model(model_name, pretrained=True, num_classes = num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lr)
scheduler = ReduceLROnPlateau(optimizer, patience = 4, factor = 0.1, threshold = 0.001)
train_dataset = CreateImageDataset(train_label_set, train_imagepath, transform=transformation)
valid_dataset = CreateImageDataset(val_label_set, val_imagepath, transform=transformation)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

params_train = {
'num_epochs':epochs,
'optimizer':optimizer,
'criterion':criterion,
'train_dl':train_dataloader,
'val_dl':valid_dataloader,
'sanity_check':False,
'lr_scheduler':scheduler,
'path2weights':'./models/weights.pt',
}

params_model = {
    'batch size':batch_size,
    'lr':lr,
    'model name':model_name,
    'pretrained':pretrained,
}

with mlflow.start_run() as run:
    mlflow.log_params(params_train)
    mlflow.log_params(params_model)
    model, train_acc, val_acc, train_loss, val_loss = training(model, params_train)

    #plot the curves
    plt.plot(train_acc, label = 'train_acc')
    plt.plot(val_acc, label = 'val_acc')
    plt.plot(train_loss, label = 'train_loss')
    plt.plot(val_loss, label = 'val_loss')
    plt.legend()
    plt.title('Accuracy and Loss Plots')
    plt.show()