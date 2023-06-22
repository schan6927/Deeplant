#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
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

def training(model, params):
    train_loss, val_loss, train_acc, val_acc =[], [], [], []
    num_epochs=params['num_epochs']
    criterion=params['criterion']
    optimizer=params['optimizer']
    train_dl=params['train_dl']
    val_dl=params['val_dl']
    scheduler=params['lr_scheduler']
    path2weights=params['path2weights']
    best_acc = 0
    for epoch in tqdm(range(num_epochs)):
        #training
        model.train()
        total_loss, total_correct = 0,0
        for x, y in tqdm(train_dl):
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            _, preds = torch.max(output, 1)
            loss = criterion(output, y.long())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += torch.sum(preds == y).item()/len(x)

        mlflow.log_metric("train loss", total_loss/len(train_dl))
        mlflow.log_metric("train accuracy", total_correct/len(train_dl))
        train_loss.append(total_loss/len(train_dl))
        train_acc.append(total_correct/len(train_dl))

        #validation
        model.eval()
        total_loss, total_correct = 0,0
        with torch.no_grad():
            for x, y in tqdm(val_dl):
                x = x.to(device)
                y = y.to(device)
                output = model(x)
                _, preds = torch.max(output, 1)
                loss = criterion(output, y.long())
                optimizer.zero_grad()
                total_loss += loss.item()
                total_correct += torch.sum(preds == y).item()/len(x)

            mlflow.log_metric("val loss", total_loss/len(val_dl))
            mlflow.log_metric("val accuracy", total_correct/len(val_dl))
            val_loss.append(total_loss/len(val_dl))
            val_acc.append(total_correct/len(val_dl))
        scheduler.step(val_loss[-1])

        #saving best model
        if val_acc[-1]>best_acc:
            best_acc = val_acc[-1]
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':val_loss[-1],
                'acc':val_acc[-1]
            }, path2weights)
        print('The Validation Loss is {} and the validation accuracy is {}'.format(val_loss[-1],val_acc[-1]))
        print('The Training Loss is {} and the training accuracy is {}'.format(train_loss[-1],train_acc[-1]))

    return model, train_acc, val_acc, train_loss, val_loss



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
'num_epochs':20,
'optimizer':optimizer,
'criterion':criterion,
'train_dl':train_dataloader,
'val_dl':valid_dataloader,
'sanity_check':False,
'lr_scheduler':scheduler,
'path2weights':'./models/weights.pt',
}

with mlflow.start_run() as run:
    mlflow.log_param("lr", lr)
    mlflow.log_param("epoch", epochs)
    mlflow.log_param("batch size", batch_size)
    model, train_acc, val_acc, train_loss, val_loss = training(model, params_train)

    #plot the curves
    plt.plot(train_acc, label = 'train_acc')
    plt.plot(val_acc, label = 'val_acc')
    plt.plot(train_loss, label = 'train_loss')
    plt.plot(val_loss, label = 'val_loss')
    plt.legend()
    plt.title('Accuracy and Loss Plots')
    plt.show()