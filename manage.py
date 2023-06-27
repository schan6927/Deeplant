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
import gc

from torch import nn
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import random_split, SubsetRandomSampler
from torch.utils.data import DataLoader
from mlflow.models.signature import infer_signature
from sklearn.model_selection import KFold

import run 
import argparse
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
import seaborn as sns

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




parser=argparse.ArgumentParser(description='training pipeline for image classification')

parser.add_argument('--model_type', default ='vit', type=str)  # 사용할 모델 선택
parser.add_argument('--model_name', default='vit_base_patch32_clip_448.laion2b_ft_in12k_in1k', type=str)  # 사용할 세부 모델 선택
parser.add_argument('--run_name', default=None, type=str)  # run 이름 선택
parser.add_argument('--sanity', default=False, type=bool)  # 빠른 test 여부

parser.add_argument('--image_size', default=448, type=int, choices=(224,448))  # 이미지 크기 재설정
parser.add_argument('--num_workers', default=4, type=int)  # 훈련에 사용할 코어 수

parser.add_argument('--epochs', default=10, type=int)  # fold당 epoch
parser.add_argument('--kfold', default=5, type=int)  # kfold 사이즈
parser.add_argument('--batch_size', default=16, type=int)  # 배치 사이즈
parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float)  # learning rate
parser.add_argument('--log_epoch', default=10, type=int)  # 몇 epoch당 기록할 지 정함
parser.add_argument('--num_classes', default=5, type=int)  # output class 개수

parser.add_argument('--factor', default=0.5, type=float)  # scheduler factor
parser.add_argument('--threshold', default=0.003, type=float)  # scheduler threshold
parser.add_argument('--momentum', default=0.9, type=float)  # optimizer의 momentum
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)  # 가중치 정규화


parser.add_argument('--data_path', default='/home/work/resized_image_datas/image_5class_5000/448/', type=str)  # data path

#parser.add_argument('--optim', default='ADAM')  # optimizer
parser.add_argument('--pretrained', default=True, type=bool, help='use pre-trained model')  # pre-train 모델 사용 여부
parser.add_argument('--load_run', default=False, type=bool, help='use runned model')  # run의 모델 사용 여부
parser.add_argument('--logged_model', default=None, type=str, help='logged model path') # 사용할 run의 path

args=parser.parse_args()



#Define data pathes
image_size = args.image_size
#homepath = '/home/work/resized_image_datas/image_5class_5000/'
datapath = args.data_path
trainpath = os.path.join(datapath,'Training')
valpath = os.path.join(datapath,'Valid')

train_label_set = pd.read_csv(f'{datapath}/train.csv')
val_label_set = pd.read_csv(f'{datapath}/valid.csv')

train_label_set['grade_encode'] = train_label_set['grade'].apply(grade_encoding)
val_label_set['grade_encode'] = val_label_set['grade'].apply(grade_encoding)

#Define kfold
fold = args.kfold

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

#Define hyperparameters
batch_size = args.batch_size
lr = args.lr
epochs = args.epochs

log_epoch = args.log_epoch
num_workers = args.num_workers
num_classes = args.num_classes

sanity = args.sanity
pretrained = args.pretrained
model_name = args.model_name

load_run = args.load_run
logged_model = args.logged_model #'runs:/523f68657d884879844be1c409bd96c0/best'

experiment_name = args.model_type
run_name = args.run_name
params_vit = {
    'num_epochs':epochs,
    'batch_size':batch_size,
    'image_size':image_size,
    'lr':lr,
    'optimizer':None,
    'loss_func':None,
    'train_dl':None,
    'val_dl':None,
    'sanity_check':sanity,
    'lr_scheduler':None,
    'log_epoch':log_epoch,
    'model_name':model_name,
    'pretrained':pretrained,
    'load_run':load_run,
    'logged_model':logged_model,
    'fold':0,
    'experiment_name':experiment_name,
    'run_name':run_name,
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
    scheduler = ReduceLROnPlateau(optimizer, patience = 2, factor = args.factor, threshold = args.threshold)

    train_sampler = SubsetRandomSampler(train_idx) 
    test_sampler = SubsetRandomSampler(val_idx)
    train_dl = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=True)

    input_schema = Schema([TensorSpec(np.dtype(np.float32),shape=(image_size,image_size))])
    output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, num_classes))])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    # train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # val_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    params_vit['optimizer']=optimizer
    params_vit['lr_scheduler']=scheduler
    params_vit['loss_func']=loss_func
    params_vit['train_dl']=train_dl
    params_vit['val_dl']=val_dl
    params_vit['fold']=fold
    run.run(model, params_vit)
    model.cpu()
    del model
    gc.collect()


torch.cuda.empty_cache()
