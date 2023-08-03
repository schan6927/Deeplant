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
import models.model as m
import models.dataset as dataset
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

parser=argparse.ArgumentParser(description='training pipeline for image classification')

parser.add_argument('--model_type', default ='vit', type=str)  # 사용할 모델 선택
parser.add_argument('--model_name', default='vit_base_patch32_clip_448.laion2b_ft_in12k_in1k', type=str)  # 사용할 세부 모델 선택
parser.add_argument('--run_name', default=None, type=str)  # run 이름 선택
parser.add_argument('--sanity', default=False, type=bool)  # 빠른 test 여부
parser.add_argument('--mode', default='train', type=str, choices=('train', 'test')) # 학습모드 / 평가모드
parser.add_argument('--image_size', default=448, type=int)  # 이미지 크기 재설정
parser.add_argument('--num_workers', default=4, type=int)  # 훈련에 사용할 코어 수

parser.add_argument('--epochs', default=10, type=int)  # fold당 epoch
parser.add_argument('--batch_size', default=16, type=int)  # 배치 사이즈
parser.add_argument('--patch_size', default=None, type=int)  # 패치 사이즈
parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float)  # learning rate
parser.add_argument('--log_epoch', default=10, type=int)  # 몇 epoch당 기록할 지 정함
parser.add_argument('--num_classes', default=5, type=int)  # output class 개수

parser.add_argument('--algorithm', default='regression', type=str, choices=('classification, regression'))  # classification, regression 중 선택
parser.add_argument('--input_columns', nargs='+', default=None, type=int) # input으로 사용할 label의 column값을 정함.
parser.add_argument('--output_columns', nargs='+', default=2, type=int) # output으로 사용할 label의 column값을 정함.
parser.add_argument('--image_column',default=0, type=int) # index로 사용할 column을 정함.
parser.add_argument('--input_shape',default=768, type=int) # fc_layer의 input shape값을 정함.
parser.add_argument('--add_graphs',nargs='+',default=None, type=str, choices=('color', 'gray', 'gcolor', 'gtexture', 'gmarbling', 'gsurface', 'gtotal')) # add할 graph 결정
parser.add_argument('--concat_graphs',nargs='+',default=None, type=str, choices=('color', 'gray', 'gcolor', 'gtexture', 'gmarbling', 'gsurface', 'gtotal')) # concat할 graph 결정

parser.add_argument('--factor', default=0.5, type=float)  # scheduler factor
parser.add_argument('--threshold', default=0.03, type=float)  # scheduler threshold
parser.add_argument('--momentum', default=0.9, type=float)  # optimizer의 momentum
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)  # 가중치 정규화


parser.add_argument('--data_path', default='/home/work/deeplant_data', type=str)  # data path
parser.add_argument('--pretrained', default=True, type=bool, help='use pre-trained model')  # pre-train 모델 사용 여부
parser.add_argument('--logged_model', default=None, type=str, help='logged model path') # 사용할 run의 path
parser.add_argument('--seed', default=42, type=int, help='random seed train/test set split') # train/test 섞을 때 seed 값
parser.add_argument('--custom_fc', default=True, type=bool, help='use custom fc') # custom fc layer 사용 여부

args=parser.parse_args()

#Define data pathes
image_size = args.image_size
datapath = args.data_path
trainpath = os.path.join(datapath,'train/cropped_448')
label_set = pd.read_csv(os.path.join(datapath,'new_train.csv'))

input_columns = args.input_columns
output_columns = args.output_columns
image_column = args.image_column
input_shape = args.input_shape
algorithm = args.algorithm
seed = args.seed
concat_graphs = args.concat_graphs
add_graphs = args.add_graphs
columns_name = label_set.columns[output_columns].values

train_set, test_set = train_test_split(label_set, test_size =0.1, random_state = seed)
train_set.reset_index(inplace=True, drop=True)
test_set.reset_index(inplace=True, drop=True)
print(train_set)
print(test_set)

#Define input transform
transformation = transforms.Compose([
transforms.Resize([image_size,image_size]),
transforms.RandomHorizontalFlip(p=0.5),
transforms.RandomVerticalFlip(p=0.5),
transforms.RandomRotation((-180,180)),
transforms.ToTensor(),
])

#Define Data loader
train_dataset = dataset.CreateImageDataset(train_set, trainpath, image_size, image_column, output_columns, input_columns, transform=transformation, train=True)
test_dataset = dataset.CreateImageDataset(test_set, trainpath, image_size, image_column, output_columns, input_columns, transform=transformation, train=False)

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
pretrained = args.pretrained
model_name = args.model_name
logged_model = args.logged_model

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
    if logged_model == None:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param('pretrained', pretrained)
    else:
        mlflow.log_param("model_name", logged_model)
        mlflow.log_param("pretrained", True)

    mlflow.log_param("num_epochs", epochs)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param('batch_size', batch_size)
    mlflow.log_param("image_size", image_size)
    mlflow.log_param("seed", seed)
    
    analyze.datasetHistogram(train_set,test_set, ['1++', '1+', '2', '3'], columns_name)
    analyze.datasetKDE(train_set,test_set, ['1++', '1+', '2', '3'], columns_name)
    
    if algorithm == 'classification':
        loss_func = nn.CrossEntropyLoss()
        train_acc_sum = np.zeros(epochs)
        val_acc_sum = np.zeros(epochs)
        train_loss_sum = np.zeros(epochs)
        val_loss_sum = np.zeros(epochs)
    elif algorithm == 'regression':
        loss_func = nn.MSELoss()
        train_mae_sum = np.zeros((epochs, num_classes))
        val_mae_sum = np.zeros((epochs, num_classes))
        train_acc_sum = np.zeros((epochs,num_classes))
        val_acc_sum = np.zeros((epochs,num_classes))
        train_loss_sum = np.zeros(epochs)
        val_loss_sum = np.zeros(epochs)
        r2_score_sum = np.zeros(epochs)
    
    mlflow.log_param("loss_func", loss_func)


#---------------모델 바꿀 시 model.py 파일과 이 부분을 바꿔주면 됨----------------------
    params_model = {
        'model_name':model_name,
        'num_classes':num_classes,
        'logged_model':logged_model,
        'pretrained':pretrained,
        'input_shape':input_shape, # input shape과 output shape은 모델 구조에 따라 잘 설정해야함.
        'output_shape':num_classes,
        'custom_fc':custom_fc,
    }
#------------------------------------------------------------------------------------

    if args.mode =='train':
        model = m.Model(params_model)
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

        train_acc_sum += train_acc
        val_acc_sum += val_acc
        train_loss_sum += train_loss
        val_loss_sum += val_loss

        if algorithm == 'regression':
            train_mae_sum += train_mae
            val_mae_sum += val_mae
            r2_score_sum += r2_score

    elif args.mode =='test':
        model = m.Model(params_model)
        model = model.to(device)

        test_dl = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
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
    
    model.cpu()
    del model
    gc.collect()

torch.cuda.empty_cache()
