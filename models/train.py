import torch
import mlflow
from tqdm import tqdm
import CM as cm
import numpy as np
import pandas as pd
import os
import sklearn
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def classification(model, params):
    num_epochs=params['num_epochs']
    loss_func=params['loss_func']
    optimizer=params['optimizer']
    scheduler=params['lr_scheduler']
    log_epoch=params['log_epoch']
    train_dl=params['train_dl']
    val_dl=params['val_dl']

    train_loss, val_loss, train_metric, val_metric =[], [], [], []
    best_acc = 0
    for epoch in tqdm(range(num_epochs)):

#         if epoch == num_epochs - 1:
#             df = pd.DataFrame(columns=['file_name', '1++', '1+', '1', '2', '3', 'score', 'prediction'])
#         else:
#             df = None
        df = None

        #training
        model.train()
        loss, metric, _ = classification_epoch(model, loss_func, train_dl, epoch,optimizer)
        mlflow.log_metric("train loss", loss, epoch)
        mlflow.log_metric("train accuracy", metric, epoch)
        train_loss.append(loss)
        train_metric.append(metric)

        #validation
        model.eval()
        with torch.no_grad():
            loss, metric, df = classification_epoch(model, loss_func, val_dl, epoch, df=df)
        mlflow.log_metric("val loss", loss, epoch)
        mlflow.log_metric("val accuracy", metric, epoch)
        val_loss.append(loss)
        val_metric.append(metric)
        scheduler.step(val_loss[-1])

        if epoch % log_epoch == log_epoch-1:
            mlflow.pytorch.log_model(model, f'epoch_{epoch}')
            
        #saving best model
        if val_metric[-1]>best_acc:
            best_acc = val_metric[-1]
            mlflow.set_tag("best", f'best at epoch {epoch}')
            mlflow.pytorch.log_model(model, f"best")
        print('The Validation Loss is {} and the validation accuracy is {}'.format(val_loss[-1],val_metric[-1]))
        print('The Training Loss is {} and the training accuracy is {}'.format(train_loss[-1],train_metric[-1]))

    if not os.path.exists('temp'):
        os.mkdir('temp')
    df.to_csv('temp/incorrect_data.csv')
    mlflow.log_artifact('temp/incorrect_data.csv')
    return model, train_metric, val_metric, train_loss, val_loss


# calculate the loss per epochs
def classification_epoch(model, loss_func, dataset_dl, epoch, fold, sanity_check=False, opt=None, df=None):
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
    
        # L1 regularization =0.001
        lambda1= 0.0000003
        l1_regularization =0.0
        for param in model.parameters():
            l1_regularization +=torch.norm(param,1)
        l1_regularization = lambda1 * l1_regularization
        
        running_loss += loss_b.item() + l1_regularization.item()
        loss_b = loss_b + l1_regularization

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
                # 틀린 이미지 csv로 저장하는 부분. 현재 소 데이터에만 적용 가능.
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
        CM.save_plot(fold)

    loss = running_loss / len_data
    metric = running_metrics / len_data
    return loss, metric, df


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
        loss, acc, _, mae = regression_epoch(model, loss_func, train_dl, epoch, num_classes, columns_name, optimizer)
        
        mlflow.log_metric("train loss", loss, epoch)
        for i in range(num_classes):
            mlflow.log_metric(f"train accuracy {columns_name[i]}",acc[i], epoch)
            mlflow.log_metric(f"train mae {columns_name[i]}",mae[i], epoch)
        train_loss.append(loss)
        train_acc.append(acc)
        train_mae.append(mae)

        #validation
        model.eval()
        with torch.no_grad():
            loss, acc, r2_score, mae = regression_epoch(model, loss_func, val_dl, epoch, num_classes, columns_name)

        mlflow.log_metric('r2 score',r2_score, epoch)
        mlflow.log_metric("val loss", loss, epoch)
        for i in range(num_classes):
            mlflow.log_metric(f"val accuracy {columns_name[i]}",acc[i], epoch)
            mlflow.log_metric(f"val mae {columns_name[i]}",mae[i], epoch)
        val_loss.append(loss)
        val_acc.append(acc)
        r2_list.append(r2_score)
        val_mae.append(mae)
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
        print('The R2 score(fixed) is {}'.format(r2_score))

    return model, train_acc, val_acc, train_loss, val_loss, r2_list, train_mae, val_mae


# calculate the loss per epochs
def regression_epoch(model, loss_func, dataset_dl, epoch, num_classes, columns_name,opt=None):
    running_loss = 0.0
    running_mae = np.zeros(num_classes)
    running_acc = np.zeros(num_classes)
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
        yb = yb.to(device)
        output = model(xb)
        
        metric_b = np.zeros(num_classes)
        total_loss = 0.0
        
        # class가 1개일 때 개별 라벨이 list 형식아니라서 for문을 못 돌림. 그래서 일단 구분함.
        if num_classes != 1:
            for i in range(num_classes):
                loss_b = loss_func(output[:, i], yb[:, i])
                total_loss += loss_b
                metric_b[i] += torch.abs(output[:,i] - yb[:,i]).sum().item()
                running_acc[i] += (torch.round(output[:,i]) == yb[:,i]).sum().item()
        else:
            loss_b = loss_func(output, yb)
            total_loss += loss_b
            metric_b += torch.abs(output - yb).sum().item()
            running_acc += (torch.round(output) == yb).sum().item()

        running_loss += total_loss.item()
        running_mae += metric_b

        if opt is not None:
            opt.zero_grad()
            total_loss.backward()
            opt.step()
            
            #---------------- 데이터 분석용 임시 추가-------------------------
            output = output.detach().cpu().numpy()
            yb = yb.cpu().numpy()

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
            #-----------------------------------------------------------------
        
        
        if opt is None:
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
            #------------------------------------------------------------------------
    
    if opt is None:
        y_mean = running_y.mean(axis=0)
        ssr = np.square(running_y - running_output).sum(axis=0)
        sst = np.square(running_y - y_mean).sum(axis=0)
        r2_score = (1 - (ssr / sst)).mean()

        if not os.path.exists('temp'):
            os.mkdir('temp')
        df.to_csv('temp/valid_output_data.csv')
        mlflow.log_artifact('temp/valid_output_data.csv', f'output_epoch_{epoch}')
        
    if opt is not None:
        if not os.path.exists('temp'):
            os.mkdir('temp')
        df.to_csv('temp/train_output_data.csv')
        mlflow.log_artifact('temp/train_output_data.csv', f'output_epoch_{epoch}')

    loss = running_loss / len_data
    mae = running_mae / len_data
    accuracy = running_acc / len_data
    return loss, accuracy, r2_score, mae


