import torch
import mlflow
from tqdm import tqdm
import CM as cm
import numpy as np
import pandas as pd
import os
import sklearn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def testing(model, params):
    num_epochs=params['num_epochs']
    test_dl=params['test_dl']
    sanity=params['sanity_check']
    log_epoch=params['log_epoch']
    signature=params['signature']
    loss_func=params['loss_func']

    test_loss, test_metric = [], []

    for epoch in tqdm(range(num_epochs)):
        if epoch == num_epochs - 1:
            df = pd.DataFrame(columns=['file_name', '1++', '1+', '1', '2', '3', 'score', 'prediction'])
        else:
            df = None

        model.eval()
        with torch.no_grad():
            loss, metric, df = loss_epoch(model, loss_func, test_dl, epoch, sanity, df=df)
        mlflow.log_metric("test loss", loss, epoch)
        mlflow.log_metric("test accuracy", metric, epoch)
        test_loss.append(loss)
        test_metric.append(metric)

        if epoch % log_epoch == log_epoch-1:
         mlflow.pytorch.log_model(model, f'model_test_epoch_{epoch}', signature=signature)

        if not os.path.exists('temp_test'):
            os.mkdir('temp_test')
        df.to_csv('temp_test/incorrect_data.csv')
        mlflow.log_artifact('temp_test/incorrect_data.csv')
        return model, test_metric, test_loss
    
def loss_epoch(model, loss_func, dataset_dl, epoch, sanity_check=False, opt=None, df=None):
    running_loss = 0.0
    running_metrics = 0.0
    len_data = len(dataset_dl.sampler)

    conf_label = []
    conf_pred = []

    for xb, yb, name_b in tqdm(dataset_dl):
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
        
        if opt is None:
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
    if opt is None:   
        new_pred = np.concatenate(conf_pred)
        new_label = np.concatenate(conf_label)
        con_mat=sklearn.metrics.confusion_matrix(new_label, new_pred)
        CM=cm.SaveCM(con_mat,epoch)
        CM.save_plot(1)

    loss = running_loss / len_data
    metric = running_metrics / len_data
    return loss, metric, df
