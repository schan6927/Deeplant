import torch
import mlflow
import os
import gc
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from torch import optim
from torch import nn

import train
import test
import analyze
import model as m
import dataset as dataset
import utils as utils
import argparse
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

parser=argparse.ArgumentParser(description='training pipeline for image classification')

parser.add_argument('--experiment_name', '--ex_name', default ='vit', type=str)  # experiment 이름 설정
parser.add_argument('--model_cfgs', default='model_cfgs.json', type=str)  # 사용할 세부 모델 
parser.add_argument('--run_name', default=None, type=str)  # run 이름 선택
parser.add_argument('--sanity', default=False, type=bool)  # 빠른 test 여부
parser.add_argument('--mode', default='train', type=str, choices=('train', 'test')) # 학습모드 / 평가모드
parser.add_argument('--num_workers', default=4, type=int)  # 훈련에 사용할 코어 수

parser.add_argument('--epochs', default=10, type=int)  # fold당 epoch
parser.add_argument('--batch_size', default=16, type=int)  # 배치 사이즈
parser.add_argument('--patch_size', default=None, type=int)  # 패치 사이즈
parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float)  # learning rate
parser.add_argument('--log_epoch', default=10, type=int)  # 몇 epoch당 기록할 지 정함

parser.add_argument('--algorithm', default='regression', type=str, choices=('classification, regression'))  # classification, regression 중 선택

parser.add_argument('--factor', default=0.5, type=float)  # scheduler factor
parser.add_argument('--threshold', default=0.03, type=float)  # scheduler threshold
parser.add_argument('--momentum', default=0.9, type=float)  # optimizer의 momentum
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)  # 가중치 정규화

parser.add_argument('--data_path', default='/home/work/deeplant_data', type=str)  # data path
parser.add_argument('--seed', default=42, type=int, help='random seed train/test set split') # train/test 섞을 때 seed 값

args=parser.parse_args()

#Define data pathes
datapath = args.data_path
trainpath = os.path.join(datapath,'train/cropped_448')
label_set = pd.read_csv(os.path.join(datapath,'new_train.csv'))
algorithm = args.algorithm
seed = args.seed

train_set, test_set = train_test_split(label_set, test_size =0.1, random_state = seed)
train_set.reset_index(inplace=True, drop=True)
test_set.reset_index(inplace=True, drop=True)
print(train_set)
print(test_set)

# Read Model's configs
with open(args.model_cfgs, 'r') as json_file:
    model_cfgs = json.load(json_file)

output_columns = model_cfgs['output_columns']
columns_name = label_set.columns[output_columns].values

#Define Data loader
train_dataset = dataset.CreateImageDataset(train_set, trainpath, model_cfgs['datasets'], output_columns, train=True)
test_dataset = dataset.CreateImageDataset(test_set, trainpath, model_cfgs['datasets'], output_columns, train=False)

#Define hyperparameters
batch_size = args.batch_size
patch_size = args.patch_size
epochs = args.epochs
lr = args.lr

log_epoch = args.log_epoch
num_workers = args.num_workers
num_classes = args.num_classes
custom_fc = args.custom_fc

sanity = args.sanity

experiment_name = args.model_type
run_name = args.run_name



# ------------------------------------------------------

mlflow.set_tracking_uri('file:///home/work/model/multi_input_model/mlruns/')
mlflow.set_experiment(experiment_name)

if run_name == None:
    run_name = experiment_name
else:
    run_name = run_name

# Start running
with mlflow.start_run(run_name=run_name) as parent_run:
    print(parent_run.info.run_id)
    mlflow.log_dict(model_cfgs)
    mlflow.log_param("num_epochs", epochs)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param('batch_size', batch_size)
    mlflow.log_param("seed", seed)
    utils.log_input(train_dataset)
    
    
    analyze.datasetHistogram(train_set,test_set, ['1++', '1+', '2', '3'], columns_name)
    analyze.datasetKDE(train_set,test_set, ['1++', '1+', '2', '3'], columns_name)
    
    if algorithm == 'classification':
        loss_func = nn.CrossEntropyLoss()
    elif algorithm == 'regression':
        loss_func = nn.MSELoss()
    
    mlflow.log_param("loss_func", loss_func)


    if args.mode =='train':
        model = m.Model(model_cfgs)
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        mlflow.log_param("total_parmas", total_params)

        optimizer = optim.Adam(model.parameters(), lr = lr)
        scheduler = ReduceLROnPlateau(optimizer, patience = 2, factor = args.factor, threshold = args.threshold)
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        params_train = {
        'num_epochs':epochs,
        'optimizer':optimizer,
        'loss_func':loss_func,
        'train_dl':train_dl,
        'val_dl':val_dl,
        'sanity_check':sanity,
        'lr_scheduler':scheduler,
        'log_epoch':log_epoch,
        'num_classes':num_classes,
        'columns_name':columns_name,
        }
        if algorithm == 'classification':
            model, train_acc, val_acc, train_loss, val_loss = train.classification(model, params_train)
        elif algorithm == 'regression':
            model, train_acc, val_acc, train_loss, val_loss, r2_score, train_mae, val_mae = train.regression(model, params_train)


    model.cpu()
    del model
    gc.collect()

torch.cuda.empty_cache()
