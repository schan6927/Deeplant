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
def classification_epoch(model, loss_func, dataset_dl, epoch, columns_name, sanity_check=False, opt=None):
    running_loss = 0.0
    len_data = len(dataset_dl.sampler)

    incorrect_output = analyze.IncorrectOutput(columns_name=["1++","1+","1","2","3"])
    confusion_matrix = analyze.ConfusionMatrix()
    accuracy = f.Accuracy(len_data, 1, "classification")

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

    train_loss, val_loss, train_acc, val_acc, r2_list, train_mae, val_mae =[], [], [], [], [], [], []
    best_loss = -1.0
    for epoch in tqdm(range(num_epochs)):
        
        #training
        model.train()
        loss, metrics = regression_epoch(model, loss_func, train_dl, epoch, num_classes, columns_name, optimizer)
        
        mlflow.log_metric("train loss", loss, epoch)
        for i in range(num_classes):
            mlflow.log_metric(f"train accuracy {columns_name[i]}",metrics[2][i], epoch)
            mlflow.log_metric(f"train mae {columns_name[i]}",metrics[1][i], epoch)
        train_loss.append(loss)
        train_acc.append(metrics[2])
        train_mae.append(metrics[1])

        #validation
        model.eval()
        with torch.no_grad():
            loss, metrics = regression_epoch(model, loss_func, val_dl, epoch, num_classes, columns_name)

        mlflow.log_metric('r2 score',metrics[0], epoch)
        mlflow.log_metric("val loss", loss, epoch)
        for i in range(num_classes):
            mlflow.log_metric(f"val accuracy {columns_name[i]}",metrics[2][i], epoch)
            mlflow.log_metric(f"val mae {columns_name[i]}",metrics[1][i], epoch)
        val_loss.append(loss)
        val_acc.append(metrics[2])
        r2_list.append(metrics[0])
        val_mae.append(metrics[1])
        scheduler.step(val_loss[-1])

        if epoch % log_epoch == log_epoch-1:
            mlflow.pytorch.log_model(model, f'model_epoch_{epoch}')
            
        #saving best model
        if val_loss[-1]<best_loss or best_loss<0.0:
            best_loss = val_loss[-1]
            mlflow.set_tag("best", f'best at epoch {epoch}')
            mlflow.pytorch.log_model(model, f"best")
            
        print('The Validation Loss is {} and the Validation Accuracy is {}'.format(val_loss[-1],val_acc[-1]))
        print('The Training Loss is {} and the Training Accuracy is {}'.format(train_loss[-1],train_acc[-1]))
        print('The Training MAE is {} and the Validation MAE is {}'.format(train_mae[-1],val_mae[-1]))
        print('The R2 score(fixed) is {}'.format(r2_list[-1]))

    return model, train_acc, val_acc, train_loss, val_loss, r2_list, train_mae, val_mae


# calculate the loss per epochs
def regression_epoch(model, loss_func, dataset_dl, epoch, num_classes, columns_name, opt=None):
    running_loss = 0.0
    len_data = len(dataset_dl.sampler)

    metrics=[]

    accuracy = f.Accuracy(len_data, num_classes, 'regression')
    r2_score = f.R2score(len_data)
    mae = f.MeanAbsError(len_data, num_classes)

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

        mae.update(output, yb)
        accuracy.update(output, yb)
        r2_score.update(output, yb)


        if opt is not None:
            opt.zero_grad()
            total_loss.backward()
            opt.step()        
        
    metrics.append(r2_score.getResult())
    metrics.append(mae.getResult())
    metrics.append(accuracy.getResult())
    output_log.saveOutputLog(epoch, opt)
 

    loss = running_loss / len_data
    return loss, metrics


