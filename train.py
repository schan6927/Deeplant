import torch
import mlflow
import numpy as np

from tqdm import tqdm
from torch.utils.data import random_split, SubsetRandomSampler
from torch.utils.data import DataLoader

from sklearn.model_selection import KFold

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# calculate the metric per mini-batch
def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects

# calculate the loss per mini-batch
def loss_batch(loss_func, output, target, opt=None):

    loss_b = loss_func(output, target)
    metric_b = metric_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()

    return loss_b.item(), metric_b

# calculate the loss per epochs
def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metrics = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in tqdm(dataset_dl):
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
        running_loss += loss_b

        if metric_b is not None:
            running_metrics += metric_b

        if sanity_check is True:
            break

    loss = running_loss / len_data
    metric = running_metrics / len_data
    return loss, metric


def training(model, params):
    num_epochs=params['num_epochs']
    loss_func=params['loss_func']
    optimizer=params['optimizer']
    scheduler=params['lr_scheduler']
    log_epoch=params['log_epoch']

    fold=params['fold']
    train_dataset=params['train_dataset']
    valid_dataset=params['valid_dataset']
    batch_size=params['batch_size']

    dataset = torch.utils.data.ConcatDataset([train_dataset, valid_dataset])
    splits = KFold(n_splits = fold, shuffle = True, random_state =42)
    
    train_loss, val_loss, train_metric, val_metric =[], [], [], []
    best_acc = 0
    
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        print('Fold {}'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx) 
        test_sampler = SubsetRandomSampler(val_idx)
        train_dl = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_dl = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
        for epoch in tqdm(range(num_epochs)):
            #training
            model.train()
            loss, metric = loss_epoch(model, loss_func, train_dl, False, optimizer)
            mlflow.log_metric("train loss", loss, epoch)
            mlflow.log_metric("train accuracy", metric, epoch)
            train_loss.append(loss)
            train_metric.append(metric)

            #validation
            model.eval()
            with torch.no_grad():
                loss, metric = loss_epoch(model, loss_func, val_dl, False)
            mlflow.log_metric("val loss", loss, epoch)
            mlflow.log_metric("val accuracy", metric, epoch)
            val_loss.append(loss)
            val_metric.append(metric)
            scheduler.step(val_loss[-1])

            if epoch % log_epoch == log_epoch-1:
                mlflow.pytorch.log_model(model, f'model_epoch_{epoch}')
                
            #saving best model
            if val_metric[-1]>best_acc:
                best_acc = val_metric[-1]
                mlflow.set_tag("description", f'best at epoch {epoch}')
                mlflow.pytorch.log_model(model, f"best")
            print('The Validation Loss is {} and the validation accuracy is {}'.format(val_loss[-1],val_metric[-1]))
            print('The Training Loss is {} and the training accuracy is {}'.format(train_loss[-1],train_metric[-1]))

    return model, train_metric, val_metric, train_loss, val_loss
