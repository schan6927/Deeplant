import torch
import mlflow
from tqdm import tqdm
import utils.analyze as analyze
import numpy as np
from torch import nn
import metric as f
import utils.analyze_regression as analyze_r

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def classification(model, params):
    num_epochs=params['num_epochs']
    loss_func=nn.CrossEntropyLoss()
    optimizer=params['optimizer']
    scheduler=params['lr_scheduler']
    log_epoch=params['log_epoch']
    train_dl=params['train_dl']
    val_dl=params['val_dl']
    columns_name=params['columns_name']
    eval_function=params['eval_function']

    train_loss, val_loss, train_metric, val_metric =[], [], [], []
    best_acc = 0
    for epoch in tqdm(range(num_epochs)):

        #training
        model.train()
        loss, metric= classification_epoch(model, loss_func, train_dl, epoch, columns_name, optimizer)
        mlflow.log_metric("train loss", loss, epoch)
        mlflow.log_metric("train accuracy", metric, epoch)
        train_loss.append(loss)
        train_metric.append(metric)

        #validation
        model.eval()
        with torch.no_grad():
            loss, metric= classification_epoch(model, loss_func, val_dl, epoch, columns_name)
        mlflow.log_metric("val loss", loss, epoch)
        mlflow.log_metric("val accuracy", metric, epoch)
        val_loss.append(loss)
        val_metric.append(metric)
        scheduler.step(val_loss[-1])

        if epoch % log_epoch == log_epoch-1:
            mlflow.pytorch.log_model(model, f'epoch_{epoch}')
            
        #saving best model
        if val_metric[-1] > best_acc:
            best_acc = val_metric[-1]
            mlflow.set_tag("best", f'best at epoch {epoch}')
            mlflow.pytorch.log_model(model, f"best")
        print('The Validation Loss is {} and the validation accuracy is {}'.format(val_loss[-1],val_metric[-1]))
        print('The Training Loss is {} and the training accuracy is {}'.format(train_loss[-1],train_metric[-1]))


    return model, train_metric, val_metric, train_loss, val_loss


# calculate the loss per epochs
def classification_epoch(model, loss_func, dataset_dl, epoch, eval_function, sanity_check=False, opt=None):
    running_loss = 0.0
    len_data = len(dataset_dl.sampler)

    incorrect_output = analyze.IncorrectOutput(columns_name=["1++","1+","1","2","3"])
    confusion_matrix = analyze.ConfusionMatrix()
    accuracy = f.Accuracy(len_data, 1, "classification")
    metrics = f.initMetrics(eval_function)

    for xb, yb, name_b in tqdm(dataset_dl):
        yb = yb.to(device).long()
        yb = yb[:,0]
        output = model(xb)
        loss_b = loss_func(output, yb)

        accuracy.update(output, yb)
    
        # L1 regularization =0.001
        lambda1= 0.0000003
        l1_regularization =0.0
        for param in model.parameters():
            l1_regularization += torch.norm(param,1)
        l1_regularization = lambda1 * l1_regularization
        
        running_loss += loss_b.item() + l1_regularization.item()
        loss_b = loss_b + l1_regularization

        if opt is not None:
            opt.zero_grad()
            loss_b.backward()
            opt.step()
    
        # Validation
        if opt is None:
            confusion_matrix.updateConfusionMatrix(output, yb)
            incorrect_output.updateIncorrectOutput(output, yb, name_b)

        if sanity_check is True:
            break

    # Validation
    if opt is None: 
        confusion_matrix.saveConfusionMatrix(epoch=epoch)
        incorrect_output.saveIncorrectOutput(filename="incorrect_output.csv", epoch=epoch)

    loss = running_loss / len_data
    metric = accuracy.getResult()
    return loss, metric


def regression(model, params):
    num_epochs=params['num_epochs']
    loss_func=nn.MSELoss()
    optimizer=params['optimizer']
    scheduler=params['lr_scheduler']
    log_epoch=params['log_epoch']
    train_dl=params['train_dl']
    val_dl=params['val_dl']
    num_classes=params['num_classes']
    columns_name=params['columns_name']
    eval_function=params['eval_function']

    train_loss, val_loss, train_acc, val_acc, r2_list, train_mae, val_mae =[], [], [], [], [], [], []
    best_loss = -1.0
    for epoch in tqdm(range(num_epochs)):
        
        #training
        model.train()
        train_loss, train_metrics = regression_epoch(model, loss_func, train_dl, epoch, num_classes, columns_name, eval_function, optimizer)
        
        mlflow.log_metric("train loss", train_loss, epoch)
        for metric in train_metrics.getMetrics():
            class_name = metric.getClassName()
            result = metric.getResult()      
            for i in range(num_classes):
                mlflow.log_metric(f"train {class_name} {columns_name[i]}", result[i], epoch)

        #validation
        model.eval()
        with torch.no_grad():
            val_loss, val_metrics = regression_epoch(model, loss_func, val_dl, epoch, num_classes, columns_name, eval_function)

        mlflow.log_metric("val loss", val_loss, epoch)
        for metric in val_metrics.getMetrics():
            class_name = metric.getClassName()
            result = metric.getResult()      
            for i in range(num_classes):
                mlflow.log_metric(f"val {class_name} {columns_name[i]}", result[i], epoch)

        scheduler.step(val_loss)

        if epoch % log_epoch == log_epoch-1:
            mlflow.pytorch.log_model(model, f'model_epoch_{epoch}')
        #saving best model
        if val_loss<best_loss or best_loss<0.0:
            best_loss = val_loss
            mlflow.set_tag("best", f'best at epoch {epoch}')
            mlflow.pytorch.log_model(model, f"best")

        print('The Training Loss is {} and the Validation Loss is {}'.format(train_loss, val_loss))
        for train_metric, val_metric in train_metrics.getMetrics(), val_metrics.getMetrics():
            print(f'The Training {train_metric.getClassName()} is {train_metric.getResult()} and the Validation {val_metric.getClassName()} is {val_metric.getResult()}')

    return model, train_acc, val_acc, train_loss, val_loss, r2_list, train_mae, val_mae


# calculate the loss per epochs
def regression_epoch(model, loss_func, dataset_dl, epoch, num_classes, columns_name, eval_function, opt=None):
    running_loss = 0.0
    len_data = len(dataset_dl.sampler)
    metrics = f.Metrics(eval_function, num_classes, 'regression', len_data)
    output_log = analyze_r.OutputLog(columns_name, num_classes)

    for xb, yb, name_b in tqdm(dataset_dl):
        yb = yb.to(device)
        output = model(xb)
        
        total_loss = 0.0
        # class가 1개일 때 개별 라벨이 list 형식아니라서 for문을 못 돌림. 그래서 일단 구분함.
        if num_classes != 1:
            for i in range(num_classes):
                loss_b = loss_func(output[:, i], yb[:, i])
                total_loss += loss_b
        else:
            loss_b = loss_func(output, yb)
            total_loss += loss_b

        running_loss += total_loss.item()
        output_log.updateOutputLog(output, yb, name_b)
        metrics.update(output, yb)

        if opt is not None:
            opt.zero_grad()
            total_loss.backward()
            opt.step()        
        
    output_log.saveOutputLog(epoch, opt)
    loss = running_loss / len_data
    return loss, metrics


