import torch
import mlflow
from tqdm import tqdm
import CM as cm
import numpy as np
import sklearn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# calculate the loss per epochs
def loss_epoch(model, loss_func, dataset_dl, epoch, fold, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metrics = 0.0
    len_data = len(dataset_dl.sampler)

    conf_label = []
    conf_pred = []

    for xb, yb in tqdm(dataset_dl):
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        loss_b = loss_func(output, yb)
        scores, pred_b = torch.max(output.data,1)
        metric_b = (pred_b == yb).sum().item()
        running_loss += loss_b.item()
        
        if opt is not None:
            opt.zero_grad()
            loss_b.backward()
            opt.step()
            predictions_conv = pred_b.cpu().numpy()
            labels_conv = yb.cpu().numpy()
            conf_pred.append(predictions_conv)
            conf_label.append(labels_conv)


        if metric_b is not None:
            running_metrics += metric_b

        if sanity_check is True:
            break

    if opt is not None:   
        new_pred = np.concatenate(conf_pred)
        new_label = np.concatenate(conf_label)

        con_mat=sklearn.metrics.confusion_matrix(new_label, new_pred)
        CM=cm.SaveCM(con_mat,epoch)
        CM.save_plot(fold)

    loss = running_loss / len_data
    metric = running_metrics / len_data
    return loss, metric


def training(model, params):
    num_epochs=params['num_epochs']
    loss_func=params['loss_func']
    optimizer=params['optimizer']
    scheduler=params['lr_scheduler']
    log_epoch=params['log_epoch']
    train_dl=params['train_dl']
    val_dl=params['val_dl']
    fold=params['fold']
    sanity=params['sanity_check']

    train_loss, val_loss, train_metric, val_metric =[], [], [], []
    best_acc = 0

    for epoch in tqdm(range(num_epochs)):
        #training
        model.train()
        loss, metric = loss_epoch(model, loss_func, train_dl, epoch, fold+1, sanity, optimizer)
        mlflow.log_metric("train loss", loss, epoch)
        mlflow.log_metric("train accuracy", metric, epoch)
        train_loss.append(loss)
        train_metric.append(metric)

        #validation
        model.eval()
        with torch.no_grad():
            loss, metric = loss_epoch(model, loss_func, val_dl, epoch, fold+1, sanity)
        mlflow.log_metric("val loss", loss, epoch)
        mlflow.log_metric("val accuracy", metric, epoch)
        val_loss.append(loss)
        val_metric.append(metric)
        scheduler.step(val_loss[-1])

        if epoch % log_epoch == log_epoch-1:
            mlflow.pytorch.log_model(model, f'model_fold_{fold+1}_epoch_{epoch}')
            
        #saving best model
        if val_metric[-1]>best_acc:
            best_acc = val_metric[-1]
            mlflow.set_tag("description", f'best at fold {fold+1}, epoch {epoch}')
            mlflow.pytorch.log_model(model, f"best")
        print('The Validation Loss is {} and the validation accuracy is {}'.format(val_loss[-1],val_metric[-1]))
        print('The Training Loss is {} and the training accuracy is {}'.format(train_loss[-1],train_metric[-1]))

    return model, train_metric, val_metric, train_loss, val_loss
