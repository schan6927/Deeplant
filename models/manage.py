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
import .models.make_model as m
import dataset as dataset
import utils as utils
import argparse
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
#-----------------------------Hard coding section-will be changed to config file----------------------------------
num_workers = 4
batch_size = 16
log_epoch=10
factor =0.5
threshold=0.03
momentum =0.9
weight_decay =5e-4
seed=42
#-----------------------------------------------------------------------------------------------------------------

parser=argparse.ArgumentParser(description='training pipeline for image classification')

parser.add_argument('--run', default ='proto', type=str)  # run 이름 설정
parser.add_argument('--name', default ='proto', type=str)  # experiment 이름 설정
parser.add_argument('--model_cfgs', default='model_cfgs.json', type=str)  # 
parser.add_argument('--mode', default='train', type=str, choices=('train', 'test')) # 학습모드 / 평가모드
parser.add_argument('--epochs', default=10, type=int)  #epochs
parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float)  # learning rate
parser.add_argument('--data_path', default='/home/work/deeplant_data', type=str)  # data path
args=parser.parse_args()

#-----------------------------------------------------------------------------------------------------------------

#Define data pathes
datapath = args.data_path
trainpath = os.path.join(datapath,'train/cropped_448')
label_set = pd.read_csv(os.path.join(datapath,'new_train.csv'))

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
epochs = args.epochs
lr = args.lr
log_epoch = args.log_epoch
experiment_name = args.name
run_name = args.run

# ------------------------------------------------------

mlflow.set_tracking_uri('file:///home/work/model/multi_input_model/mlruns/')
mlflow.set_experiment(experiment_name)

# Start running
with mlflow.start_run(run_name=run_name) as parent_run:
    print(parent_run.info.run_id)
    mlflow.log_dict(model_cfgs, 'config/configs.json')
    mlflow.log_param("num_epochs", epochs)
    mlflow.log_param("learning_rate", lr)

    if args.mode =='train':
        model = m.create_model(model_cfgs)
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        mlflow.log_param("total_parmas", total_params)

        optimizer = optim.Adam(model.parameters(), lr = lr)
        scheduler = ReduceLROnPlateau(optimizer, patience = 2, factor = factor, threshold = threshold)
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        params_train = {
        'num_epochs':epochs,
        'optimizer':optimizer,
        'train_dl':train_dl,
        'val_dl':val_dl,
        'lr_scheduler':scheduler,
        'log_epoch':log_epoch,
        'num_classes':len(model_cfgs['output_columns']),
        'columns_name':columns_name
        }
        
        algorithm = model.algorithm()
        
        if algorithm == 'classification':
             model, train_acc, val_acc, train_loss, val_loss = train.classification(model, params_train)
        elif algorithm == 'regression':
             model, train_acc, val_acc, train_loss, val_loss, r2_score, train_mae, val_mae = train.regression(model, params_train)

    model.cpu()
    del model
    gc.collect()

torch.cuda.empty_cache()
