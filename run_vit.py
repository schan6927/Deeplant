import torch
import pandas as pd
import numpy as np

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
from torch.utils.data import random_split, SubsetRandomSampler
from torch.utils.data import DataLoader

from sklearn.model_selection import KFold

import vit 
from datetime import datetime as dt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


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
image_size = 224
homepath = '/home/work/resized_image_datas/image_5class_5000/'
datapath = os.path.join(homepath,str(image_size))
trainpath = os.path.join(datapath,'Training')
valpath = os.path.join(datapath,'Valid')

train_label_set = pd.read_csv(f'{datapath}/train.csv')
val_label_set = pd.read_csv(f'{datapath}/valid.csv')

train_label_set['grade_encode'] = train_label_set['grade'].apply(grade_encoding)
val_label_set['grade_encode'] = val_label_set['grade'].apply(grade_encoding)

#Define kfold
fold = 5

#Define input transform
transformation = transforms.Compose([
transforms.RandomHorizontalFlip(p=0.3),
transforms.RandomVerticalFlip(p=0.3),
transforms.RandomRotation((-20,20)),
transforms.ToTensor(),
])

#Define Data loader
train_dataset = CreateImageDataset(train_label_set, trainpath, transform=transformation)
valid_dataset = CreateImageDataset(val_label_set, valpath, transform=transformation)
dataset = torch.utils.data.ConcatDataset([train_dataset, valid_dataset])
splits = KFold(n_splits = fold, shuffle = True, random_state =42)
#train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

#Define hyperparameters
batch_size = 16
lr = 0.0001
epochs = 10

log_epoch = 5
num_workers = 4
num_classes = 5
pretrained = True
model_name = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'

load_run = False
logged_model = 'runs:/523f68657d884879844be1c409bd96c0/best'

params_vit = {
    'num_epochs':epochs,
    'batch_size':batch_size,
    'image_size':image_size,
    'lr':lr,
    'optimizer':None,
    'loss_func':None,
    'train_dl':None,
    'val_dl':None,
    'sanity_check':False,
    'lr_scheduler':None,
    'log_epoch':log_epoch,
    'model_name':model_name,
    'pretrained':pretrained,
    'load_run':load_run,
    'logged_model':logged_model,
    'fold':0,
}

for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):            
    print('Fold {}'.format(fold + 1))

    if load_run == True:
        model = mlflow.pytorch.load_model(logged_model)
    else:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes = num_classes)

    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    scheduler = ReduceLROnPlateau(optimizer, patience = 4, factor = 0.1, threshold = 0.001)

    train_sampler = SubsetRandomSampler(train_idx) 
    test_sampler = SubsetRandomSampler(val_idx)
    train_dl = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_dl = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    params_vit['optimizer']=optimizer
    params_vit['lr_scheduler']=scheduler
    params_vit['loss_func']=loss_func
    params_vit['train_dl']=train_dl
    params_vit['val_dl']=val_dl
    params_vit['fold']=fold
    vit.run(model, params_vit)