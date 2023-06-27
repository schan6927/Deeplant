#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import mlflow
import train 
from datetime import datetime as dt

def run(model, params):
    params_train = {
    'num_epochs':params['num_epochs'],
    'optimizer':params['optimizer'],
    'loss_func':params['loss_func'],
    'train_dl':params['train_dl'],
    'val_dl':params['val_dl'],
    'sanity_check':params['sanity_check'],
    'lr_scheduler':params['lr_scheduler'],
    'log_epoch':params['log_epoch'],
    'fold':params['fold'],
    'signature':params['signature']
    }
    experiment_name = params['experiment_name']
    mlflow.set_experiment(experiment_name)
    
    now = dt.now()
    date_time_string = now.strftime("%Y-%m-%d %H:%M:%S")

    if params['run_name'] == None:
        run_name = f"{experiment_name} fold {params['fold'] + 1}: " + str(date_time_string)
    else:
        run_name = f"{params['run_name']} fold {params['fold'] + 1}: " + str(date_time_string)

    with mlflow.start_run(run_name=run_name) as run:
        print(run.info.run_id)
        if params['load_run'] == False:
            mlflow.log_param("model_name", params['model_name'])
            mlflow.log_param('pretrained', params['pretrained'])
        else:
            mlflow.log_param("model_name", params['logged_model'])
            mlflow.log_param("pretrained", True)

        mlflow.log_param("num_epochs", params['num_epochs'])
        mlflow.log_param("learning_rate", params['lr'])
        mlflow.log_param('batch_size', params['batch_size'])
        mlflow.log_param("image_size", params['image_size'])

        mlflow.log_param("optimizer", params['optimizer'])
        mlflow.log_param("loss_func", params['loss_func'])
        mlflow.log_param("lr_scheduler", params['lr_scheduler'])
        
        model, train_acc, val_acc, train_loss, val_loss = train.training(model, params_train)

        #plot the curves
        plt.plot(train_acc, label = 'train_acc')
        plt.plot(val_acc, label = 'val_acc')
        plt.plot(train_loss, label = 'train_loss')
        plt.plot(val_loss, label = 'val_loss')
        plt.legend()
        plt.title('Accuracy and Loss Plots')
        plt.show()
        
        mlflow.log_figure(plt, "Graph_"+str(params['log_epoch'])+'_'+str(params['fold+1'])+'.jpg')
        plt.clf()


