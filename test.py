import torch
import mlflow
from tqdm import tqdm
import CM as cm
import numpy as np
import pandas as pd
import os
import sklearn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def classification(model, params):
    num_epochs=params['num_epochs']
    test_dl=params['test_dl']
    sanity=params['sanity_check']
    loss_func=params['loss_func']

    test_loss, test_metric = [], []

    for epoch in tqdm(range(num_epochs)):
        df = pd.DataFrame(columns=['file_name', '1++', '1+', '1', '2', '3', 'score', 'prediction'])
        model.eval()
        with torch.no_grad():
            loss, metric, df = classification_epoch(model, loss_func, test_dl, epoch, sanity, df=df)
        mlflow.log_metric("test loss", loss, epoch)
        mlflow.log_metric("test accuracy", metric, epoch)
        test_loss.append(loss)
        test_metric.append(metric)

        if not os.path.exists('temp_test'):
            os.mkdir('temp_test')
        df.to_csv('temp_test/incorrect_data.csv')
        mlflow.log_artifact('temp_test/incorrect_data.csv',f'incorrect_data_epoch_{epoch}')
        return model, test_metric, test_loss

def classification_epoch(model, loss_func, dataset_dl, epoch, sanity_check=False, df=None):
    running_loss = 0.0
    running_metrics = 0.0
    len_data = len(dataset_dl.sampler)

    conf_label = []
    conf_pred = []

    for xb, yb, name_b in tqdm(dataset_dl):
        xb = xb.to(device)
        yb = yb.to(device)
        yb = yb[:,0]
        output = model(xb)
        loss_b = loss_func(output, yb)
        scores, pred_b = torch.max(output.data,1)
        metric_b = (pred_b == yb).sum().item()
        running_loss += loss_b.item()
        
        # Confusion Matrix에 쓰일 data append하는 부분
        predictions_conv = pred_b.cpu().numpy()
        labels_conv = yb.cpu().numpy()
        conf_pred.append(predictions_conv)
        conf_label.append(labels_conv)

        if df is not None:
            # 틀린 이미지 csv로 저장하는 부분
            index = torch.nonzero((pred_b != yb)).squeeze().cpu().tolist()
            if not isinstance(index, list):
                index = [index]  # index가 단일 값인 경우에 리스트로 변환하여 처리
            scores = scores.cpu().numpy()
            output = list(output.detach().cpu().numpy())
            name_b = list(name_b)
            for i in index:
                data = {'file_name':name_b[i], 
                        '1++':output[i][0],
                        '1+':output[i][1],
                        '1':output[i][2],
                        '2':output[i][3],
                        '3':output[i][4], 
                        'score':scores[i], 
                        'prediction':predictions_conv[i]}
                new_row = pd.DataFrame(data=data, index=['file_name'])
                df = pd.concat([df,new_row], ignore_index=True)


        if metric_b is not None:
            running_metrics += metric_b

        if sanity_check is True:
            break

    # Validation 일 때 Confusion Matrix 그리는 부분 
    new_pred = np.concatenate(conf_pred)
    new_label = np.concatenate(conf_label)
    con_mat=sklearn.metrics.confusion_matrix(new_label, new_pred)
    CM=cm.SaveCM(con_mat,epoch)
    CM.save_plot(1)

    loss = running_loss / len_data
    metric = running_metrics / len_data
    return loss, metric, df


def regression(model, params):
    num_epochs=params['num_epochs']
    loss_func=params['loss_func']
    test_dl=params['test_dl']
    sanity=params['sanity_check']
    num_classes=params['num_classes']

    test_loss, test_metric=[], []
    for epoch in tqdm(range(num_epochs)):
        #testing
        model.eval()
        with torch.no_grad():
            loss, metric = regression_epoch(model, loss_func, test_dl, epoch, num_classes, sanity)
        mlflow.log_metric("val loss", loss, epoch)

        for i in range(num_classes):
            mlflow.log_metric(f"val metric {i}",metric[i], epoch)
        test_loss.append(loss)
        test_metric.append(metric)   

        print('The Testing Loss is {} and the Testing metric is {}'.format(test_loss[-1],test_metric[-1]))

    return model, test_metric, test_loss


# calculate the loss per epochs
def regression_epoch(model, loss_func, dataset_dl, epoch, num_classes, sanity_check=False):
    running_loss = 0.0
    running_metrics = np.zeros(num_classes)
    len_data = len(dataset_dl.sampler)
    df = pd.DataFrame(columns=['file_name','predict1','predict2','regression1','regression2'])
    for xb, yb, name_b in tqdm(dataset_dl):
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        
        metric_b = np.zeros(num_classes)
        total_loss = 0.0
        for i in range(num_classes):
            loss_b = loss_func(output[:, i], yb[:, i])
            total_loss += loss_b
            metric_b[i] += loss_b.item()
            
            # Tesiting값 저장하는 부분
            output = list(output.detach().cpu().numpy())
            yb = list(yb.cpu().numpy())
            name_b = list(name_b)
            for i in range(len(output)):
                data = {'file_name':name_b[i],
                        'predict1':output[i][0], 
                        'predict2':output[i][1],
                        'regression1':yb[i][0],
                        'regression2':yb[i][1],
                       }
                new_row = pd.DataFrame(data=data, index=['file_name'])
                df = pd.concat([df,new_row], ignore_index=True)

            if not os.path.exists('temp'):
                os.mkdir('temp')
            df.to_csv('temp/last_data.csv')
            mlflow.log_artifact('temp/last_data.csv', f'output_epoch_{epoch}')

        running_loss += total_loss.item()
        if metric_b is not None:
            running_metrics += metric_b

        if sanity_check is True:
            break

    loss = running_loss / len_data
    metric = running_metrics / len_data
    return loss, metric
