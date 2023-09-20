import pandas as pd
import numpy as np
import sklearn
import torch
import os
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt


class IncorrectOutput():
    def __init__(self, columns_name):
        #-------------결과 저장할 data frame 정의하는 곳-------------
        # 여기 바꾸면 아래 validation 저장하는 부분도 바꿔야함.
        self.columns_name = columns_name
        columns = ['file_name']
        for i in range(columns_name):
            columns.append(columns_name[i])
        columns.append("predict")
        self.df = pd.DataFrame(columns=columns)
        #----------------------------------------------------------

    def updateIncorrectOutput(self, pred_b, yb, name_b, scores, output):
        # 틀린 이미지 csv로 저장하는 부분. 현재 소 데이터에만 적용 가능.
        index = torch.nonzero((pred_b != yb)).squeeze().cpu().tolist()
        if not isinstance(index, list):
            index = [index]  # index가 단일 값인 경우에 리스트로 변환하여 처리
        pred_b = pred_b.cpu().numpy()
        scores = scores.cpu().numpy()
        output = list(output.detach().cpu().numpy())
        name_b = list(name_b)

        for i in index:
            data = {'file_name':name_b[i]}
            # class 개수 1개면 문제 생겨서 나눔.
            if len(output[0]) != 1:
                for j in range(len(output[0])):
                    data[self.columns_name[j]] = output[i][j]
            else:
                data[self.columns_name[0]] = output[i]
            data['score'] = scores[i]
            data['predict'] = pred_b[i]
            new_row = pd.DataFrame(data=data, index=['file_name'])
            df = pd.concat([df,new_row], ignore_index=True)

    def saveIncorrectOutput(self, filename, epoch):
        if not os.path.exists('temp'):
            os.mkdir('temp')
        self.df.to_csv(f'temp/{filename}.csv')
        mlflow.log_artifact(f'temp/{filename}.csv', f'output_epoch_{epoch}')



class ConfusionMatrix():
    def __init__(self):
        self.conf_pred = []
        self.conf_label = []


    def updateConfusionMatrix(self, pred_b, yb):
        predictions_conv = pred_b.cpu().numpy()
        labels_conv = yb.cpu().numpy()
        self.conf_pred.append(predictions_conv)
        self.conf_label.append(labels_conv)


    def saveConfusionMatrix(self, epoch):
        new_pred = np.concatenate(self.conf_pred)
        new_label = np.concatenate(self.conf_label)
        con_mat=sklearn.metrics.confusion_matrix(new_label, new_pred)

        cfs=sns.heatmap(con_mat,annot=True)
        cfs.set(title='Confusion Matrix', ylabel='True lable', xlabel='Predict label')
        figure = cfs.get_figure()
        
        # Save the plot as an image
        mlflow.log_figure(figure, f"output_epoch_{epoch}/confusion_matrix.jpg")
        
        #close figure
        plt.clf()
