import torch
import pandas as pd
import numpy as np
import math

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

import train
import test
import argparse
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from datetime import datetime as dt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class CreateImageDataset(Dataset):
    def __init__(self, labels, img_dir, index, columns, algorithm, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_labels = labels
        self.index = index
        self.columns = columns
        self.algorithm = algorithm

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if self.algorithm == 'classification':
            label = torch.tensor(self.img_labels.iloc[idx, self.columns])
        elif self.algorithm == 'regression':
            label = torch.tensor(self.img_labels.iloc[idx, self.columns], dtype=torch.float32)
        name = self.img_labels.iloc[idx, self.index]
        img_path = os.path.join(self.img_dir, 'cow_data')
        img_path = os.path.join(img_path, name)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, name



parser=argparse.ArgumentParser(description='training pipeline for image classification')

parser.add_argument('--model_type', default ='vit', type=str)  # 사용할 모델 선택
parser.add_argument('--model_name', default='vit_base_patch32_clip_448.laion2b_ft_in12k_in1k', type=str)  # 사용할 세부 모델 선택
parser.add_argument('--run_name', default=None, type=str)  # run 이름 선택
parser.add_argument('--sanity', default=False, type=bool)  # 빠른 test 여부
parser.add_argument('--mode', default='train', type=str, choices=('train', 'test')) # 학습모드 / 평가모드
parser.add_argument('--image_size', default=448, type=int, choices=(224,448))  # 이미지 크기 재설정
parser.add_argument('--num_workers', default=4, type=int)  # 훈련에 사용할 코어 수

parser.add_argument('--epochs', default=10, type=int)  # fold당 epoch
parser.add_argument('--kfold', default=5, type=int)  # kfold 사이즈
parser.add_argument('--batch_size', default=16, type=int)  # 배치 사이즈
parser.add_argument('--patch_size', default=None, type=int)  # 패치 사이즈
parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float)  # learning rate
parser.add_argument('--log_epoch', default=10, type=int)  # 몇 epoch당 기록할 지 정함
parser.add_argument('--num_classes', default=5, type=int)  # output class 개수

parser.add_argument('--algorithm', default='classification', type=str, choices=('classification, regression'))  # classification, regression 중 선택
parser.add_argument('--columns', nargs='+', default=2, type=int) # 사용할 label의 column값을 정함.
parser.add_argument('--index',default=0, type=int) # index로 사용할 column을 정함.

parser.add_argument('--factor', default=0.5, type=float)  # scheduler factor
parser.add_argument('--threshold', default=0.003, type=float)  # scheduler threshold
parser.add_argument('--momentum', default=0.9, type=float)  # optimizer의 momentum
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)  # 가중치 정규화


parser.add_argument('--data_path', default='/home/work/original_cropped_image_dataset/image_5class_6000/448/', type=str)  # data path
parser.add_argument('--pretrained', default=True, type=bool, help='use pre-trained model')  # pre-train 모델 사용 여부
parser.add_argument('--logged_model', default=None, type=str, help='logged model path') # 사용할 run의 path

args=parser.parse_args()

#Define data pathes
image_size = args.image_size
datapath = args.data_path
trainpath = os.path.join(datapath,'Training')
train_label_set = pd.read_csv(os.path.join(datapath,'train.csv'))

columns = args.columns
index = args.index
algorithm = args.algorithm

print(train_label_set)

#Define kfold
kfold = args.kfold

#Define input transform
transformation = transforms.Compose([
transforms.RandomHorizontalFlip(p=0.3),
transforms.RandomVerticalFlip(p=0.3),
transforms.RandomRotation((-20,20)),
transforms.ToTensor(),
])

#Define Data loader
dataset = CreateImageDataset(train_label_set, trainpath, index, columns, algorithm, transform=transformation)
splits = KFold(n_splits = kfold, shuffle = True, random_state =42)

#Define hyperparameters
batch_size = args.batch_size
patch_size = args.patch_size
epochs = args.epochs
lr = args.lr

log_epoch = args.log_epoch
num_workers = args.num_workers
num_classes = args.num_classes

# +
sanity = args.sanity
pretrained = args.pretrained
model_name = args.model_name

load_run = args.load_run
logged_model = args.logged_model
# -

experiment_name = args.model_type
run_name = args.run_name

columns_name = train_label_set.columns[columns]
if not isinstance(columns_name, list):
    columns_name = [columns_name]  # columns_name이 단일 값인 경우에 리스트로 변환하여 처리
#------------------------------------------------------

mlflow.set_experiment(experiment_name)
now = dt.now()
date_time_string = now.strftime("%Y-%m-%d %H:%M:%S")


if run_name == None:
    run_name = experiment_name
else:
    run_name = run_name

# Start running
with mlflow.start_run(run_name=run_name) as parent_run:
    print(parent_run.info.run_id)
    if load_run == False:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param('pretrained', pretrained)
    else:
        mlflow.log_param("model_name", logged_model)
        mlflow.log_param("pretrained", True)

    mlflow.log_param("num_epochs", epochs)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param('batch_size', batch_size)
    mlflow.log_param("image_size", image_size)

    if algorithm == 'classification':
        loss_func = nn.CrossEntropyLoss()
        train_acc_sum = np.zeros(epochs)
        val_acc_sum = np.zeros(epochs)
        train_loss_sum = np.zeros(epochs)
        val_loss_sum = np.zeros(epochs)
    elif algorithm == 'regression':
        loss_func = nn.MSELoss()
        train_acc_sum = np.zeros((epochs,num_classes))
        val_acc_sum = np.zeros((epochs,num_classes))
        train_loss_sum = np.zeros(epochs)
        val_loss_sum = np.zeros(epochs)
        r2_score_sum = np.zeros(epochs)
    
    mlflow.log_param("loss_func", loss_func)

    if args.mode =='train':
        for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):            
            print('Fold {}'.format(fold + 1))

            if logged_model is not None:
                model = mlflow.pytorch.load_model(logged_model)
            else:
                model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, exportable=True)

            total_params = sum(p.numel() for p in model.parameters())
            mlflow.log_param("total_parmas", total_params)

            
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr = lr)
            scheduler = ReduceLROnPlateau(optimizer, patience = 2, factor = args.factor, threshold = args.threshold)

            train_sampler = SubsetRandomSampler(train_idx) 
            test_sampler = SubsetRandomSampler(val_idx)
            train_dl = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
            val_dl = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=True)

            input_schema = Schema([TensorSpec(np.dtype(np.float32),shape=(image_size,image_size))])
            output_schema = Schema([TensorSpec(np.dtype(np.float32), (1, num_classes))])
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)

            params_train = {
            'num_epochs':epochs,
            'optimizer':optimizer,
            'loss_func':loss_func,
            'train_dl':train_dl,
            'val_dl':val_dl,
            'sanity_check':sanity,
            'lr_scheduler':scheduler,
            'log_epoch':log_epoch,
            'fold':fold,
            'signature':signature,
            'num_classes':num_classes,
            'columns_name':columns_name,
            }

            with mlflow.start_run(run_name=str(fold+1), nested=True) as run:
                print(run.info.run_id)
                if algorithm == 'classification':
                    model, train_acc, val_acc, train_loss, val_loss = train.classification(model, params_train)
                elif algorithm == 'regression':
                    model, train_acc, val_acc, train_loss, val_loss, r2_score = train.regression(model, params_train)
            
                train_acc_sum += train_acc
                val_acc_sum += val_acc
                train_loss_sum += train_loss
                val_loss_sum += val_loss

                if algorithm == 'regression':
                    r2_score_sum += r2_score

                #plot the curves
                plt.plot(train_acc, label = 'train_acc')
                plt.plot(val_acc, label = 'val_acc')
                plt.plot(train_loss, label = 'train_loss')
                plt.plot(val_loss, label = 'val_loss')
                plt.legend()
                plt.title('Accuracy and Loss Plots')
                figure = plt.gcf()
                mlflow.log_figure(figure, "Graph_"+str(log_epoch)+'_'+str(fold+1)+'.jpg')
                plt.clf()

        for i in range(epochs):
            mlflow.log_metric("train loss", train_loss_sum[i] / kfold, i)
            if algorithm == 'classification':
                mlflow.log_metric("train accuracy", train_acc_sum[i] / kfold , i)
                mlflow.log_metric("val accuracy", val_acc_sum[i] / kfold , i)
            elif algorithm == 'regression':
                if num_classes != 1:
                    for j in range(num_classes):
                        mlflow.log_metric(f"train {columns_name[j]}",train_acc_sum[i][j]/ kfold , i)
                        mlflow.log_metric(f"val {columns_name[j]}",val_acc_sum[i][j]/ kfold , i)
                else:
                    mlflow.log_metric(f"train {columns_name[0]}",train_acc_sum[i]/ kfold , i)
                    mlflow.log_metric(f"val {columns_name[0]}",val_acc_sum[i]/ kfold , i)
                mlflow.log_metric('r2 score', r2_score_sum[i] / kfold, i)

            mlflow.log_metric("val loss", val_loss_sum[i] / kfold, i)
  

    elif args.mode =='test':
        if logged_model is not None:
            model = mlflow.pytorch.load_model(logged_model)
        else:
            model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, exportable=True)
        model = model.to(device)

        test_dl = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        input_schema = Schema([TensorSpec(np.dtype(np.float32),shape=(image_size,image_size))])
        output_schema = Schema([TensorSpec(np.dtype(np.float32), (1, num_classes))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        params_test ={
            'num_epochs':epochs,
            'test_dl':test_dl,
            'sanity_check':sanity,
            'loss_func':loss_func,
            'num_classes':num_classes,
            'columns_name':columns_name,
        }

        with mlflow.start_run(run_name='Test', nested=True) as run:
            if algorithm == 'classification':
                model, test_acc, test_loss = test.classification(model, params_test)
            elif algorithm == 'regression':
                model, test_acc, test_loss, r2_score = test.regression(model, params_test)

            plt.plot(test_acc, label = 'test_acc')
            plt.plot(test_loss, label = 'test_loss')
            plt.legend()
            plt.title('Accuracy and Loss Plots')
            figure = plt.gcf()
            mlflow.log_figure(figure, "Graph_"+str(log_epoch)+'_'+'test'+'.jpg')
            plt.clf()

            for i in range(epochs):
                mlflow.log_metric("test loss", test_loss[i], i)
                if algorithm == 'classification':
                    mlflow.log_metric("test accuracy", test_acc[i], i)
                elif algorithm == 'regression':
                    if num_classes != 1:
                        for j in range(num_classes):
                            mlflow.log_metric(f"test {columns_name[j]}",test_acc[i][j]/ kfold , i)
                    else:
                        mlflow.log_metric(f"test {columns_name[0]}",test_acc[i]/ kfold , i)
                    mlflow.log_metric('r2 score', r2_score[i] / kfold, i)
    
    model.cpu()
    del model
    gc.collect()

torch.cuda.empty_cache()
