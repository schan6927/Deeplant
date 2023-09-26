import torch
import mlflow
from tqdm import tqdm
import models.utils.confusion_matrix as cm
import numpy as np
import pandas as pd
import os
import sklearn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def classification(model, params):
    """    
    모델을 평가모드로 전환시킨 후, (loss, metric, df)를 classification_epoch()를 통해 구한다.
    그 후, loss 와 metric을 각각 test_loss 와 test_metric에 추가해준 후, test_loss 와 test_metric 을 반환한다. 
    
    (
    model: 사용할 모델
    ,params: 실험을 진행할 때 주어진 명령들의 집합
    )
    """
    num_epochs=params['num_epochs']
    test_dl=params['test_dl']
    sanity=params['sanity_check']
    loss_func=params['loss_func']

    test_loss, test_metric = [], []

    for epoch in tqdm(range(num_epochs)):
        #df = pd.DataFrame(columns=['file_name', '1++', '1+', '1', '2', '3', 'score', 'prediction'])
        df = None
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
    """
    학습을 거친 후 손실값, 정확도를 반환하고, 혼동행렬을 작성한다.
    
    모델에 입력값을 넣은 후, 손실함수에 결과값과 원본 데이터를 넣은 값인 loss 값을 모두 더한 후, 데이터의 길이 만큼 나눈 값을 loss에 저장한다.
    또한 실제값과 결과값 사이에서 정확히 예측한 수를 metric에 더한후, 데이터의 길이 만큼 나눈 값을 metric에 저장한다.
    결과 값을 Confusion Matrix에 쓰일 결과값과 예측값을 SaveCM()를 통해 confusion matrix를 작성한다.
    최종적으로 구해진 loss, metric, df를 반환한다.
    
    (
    model: 사용할 모델
    ,loss_func: 사용할 손실 함수
    ,dataset_dl: 사용할 데이터셋의 데이터로더
    ,epoch: 현재 학습 횟수
    ,sanity_check: 결함 여부 나타내는 flag
    ,df: 실험에 따라 추가할 데이터 프레임(default = None)
    )
    """
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
    cm.SaveCM(con_mat, 1, epoch)

    loss = running_loss / len_data
    metric = running_metrics / len_data
    return loss, metric, df


def regression(model, params):
    """    
    모델을 평가모드로 전환시킨 후, (loss, metric, df)를 classification_epoch()를 통해 구한다.
    그 후, loss 와 metric을 각각 test_loss 와 test_metric에 추가해준 후, test_loss 와 test_metric 을 반환한다. 
    
    (
    model: 사용할 모델
    ,params: 실험을 진행할 때 주어진 명령들의 집합
    )
    """
    
    num_epochs=params['num_epochs']
    loss_func=params['loss_func']
    test_dl=params['test_dl']
    sanity=params['sanity_check']
    num_classes=params['num_classes']
    columns_name=params['columns_name']

    test_loss, test_metric, r2_list =[], [], []
    for epoch in tqdm(range(num_epochs)):
        #testing
        model.eval()
        with torch.no_grad():
            loss, metric, r2_score = regression_epoch(model, loss_func, test_dl, epoch, num_classes, columns_name, sanity)
        mlflow.log_metric("val loss", loss, epoch)

        for i in range(num_classes):
            mlflow.log_metric(f"val {columns_name}",metric[i], epoch)
        test_loss.append(loss)
        test_metric.append(metric)
        r2_list.append(r2_score)   

        print('The Testing Loss is {} and the Testing metric is {}'.format(test_loss[-1],test_metric[-1]))
        print('The R2 score(fixed) is {}'.format(r2_score))

    return model, test_metric, test_loss, r2_list


# calculate the loss per epochs
def regression_epoch(model, loss_func, dataset_dl, epoch, num_classes, columns_name, sanity_check=False):
    """
    학습을 거친 후 손실값, 정확도를 반환하고, 혼동행렬을 작성한다.
    
    모델에 입력값을 넣은 후, 손실함수에 결과값과 원본 데이터를 넣은 값인 loss 값을 모두 더한 후, 데이터의 길이 만큼 나눈 값을 loss에 저장한다.
    또한 실제값과 결과값 사이에서 정확히 예측한 수를 metric에 더한후, 데이터의 길이 만큼 나눈 값을 metric에 저장한다.
    결과 값을 Confusion Matrix에 쓰일 결과값과 예측값을 SaveCM()를 통해 confusion matrix를 작성한다.
    최종적으로 구해진 loss, metric, df를 반환한다.
    
    (
    model: 사용할 모델
    ,loss_func: 사용할 손실 함수
    ,dataset_dl: 사용할 데이터셋의 데이터로더
    ,epoch: 현재 학습 횟수
    ,num_classes: 데이터의 클래스의 개수
    ,columns_name: 각 클래스 컬럼의 이름
    ,sanity_check: 결함 여부 나타내는 flag
    ,df: 실험에 따라 추가할 데이터 프레임(default = None)
    )
    """
    
    running_loss = 0.0
    running_metrics = np.zeros(num_classes)
    running_y = None
    running_output = None
    len_data = len(dataset_dl.sampler)
    r2_score =0.0

    #-------------결과 저장할 data frame 정의하는 곳-------------
    # 여기 바꾸면 아래 validation 저장하는 부분도 바꿔야함.
    columns = ['file_name']
    for i in range(num_classes):
        columns.append('predict ' + columns_name[i])
        columns.append('label ' + columns_name[i])
    df = pd.DataFrame(columns=columns)
    #----------------------------------------------------------

    for xb, yb, name_b in tqdm(dataset_dl):
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        
        metric_b = np.zeros(num_classes)
        total_loss = 0.0
        
        # class가 1개일 때 개별 라벨이 list 형식아니라서 for문을 못 돌림. 그래서 일단 구분함.
        if num_classes != 1:
            for i in range(num_classes):
                loss_b = loss_func(output[:, i], yb[:, i])
                total_loss += loss_b
                metric_b[i] += loss_b.item()
        else:
            loss_b = loss_func(output, yb)
            total_loss += loss_b
            metric_b += loss_b.item()
            
        #-------------------- validation값 저장하는 부분 -------------------------
        # 여기 바꾸면 위에 data frame 정의하는 곳도 바꿔야함. 
        output = output.detach().cpu().numpy()
        yb = yb.cpu().numpy()
        
        if running_y is None:
            running_y = np.array(yb)
        else:
            running_y = np.vstack((running_y,yb))
            
        if running_output is None:
            running_output = np.array(output)
        else:
            running_output = np.vstack((running_output,output))
        
        output = list(output)
        yb = list(yb)
        name_b = list(name_b)
        for i in range(len(output)):
            data = {'file_name':name_b[i]}
            # class 개수 1개면 문제 생겨서 나눔.
            if num_classes != 1:
                for j in range(num_classes):
                    data['predict ' + columns_name[j]] = output[i][j]
                    data['label ' + columns_name[j]] = yb[i][j]
            else:
                data['predict ' + columns_name[0]] = output[i]
                data['label ' + columns_name[0]] = yb[i]
            new_row = pd.DataFrame(data=data, index=['file_name'])
            df = pd.concat([df,new_row], ignore_index=True)
            
        if not os.path.exists('temp'):
            os.mkdir('temp')
        df.to_csv('temp/last_data.csv')
        mlflow.log_artifact('temp/last_data.csv', f'output_epoch_{epoch}')
        #------------------------------------------------------------------------

        running_loss += total_loss.item()
        if metric_b is not None:
            running_metrics += metric_b

        if sanity_check is True:
            break

    n = len(running_y)
    print("n:",n)
    y_mean = running_y.mean(axis=0)
    print("y_mean:",y_mean)
    ssr = np.square(running_y - running_output).sum(axis=0)
    print("ssr:",ssr)
    sst = np.square(running_y - y_mean).sum(axis=0)
    print("sst:",sst)
    print("1 - (ssr / sst):",1 - (ssr / sst))
    r2_score = (1 - (ssr / sst)).mean()
    print("r2_score:",r2_score)

    loss = running_loss / len_data
    metric = running_metrics / len_data
    return loss, metric , r2_score
