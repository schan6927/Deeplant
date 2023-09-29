import pandas as pd
import os
import mlflow
class OutputLog():
    def __init__(self, columns_name, num_classes):
        self.num_classes = num_classes
        self.columns_name = columns_name
        columns = ['file_name']
        for i in range(num_classes):
            columns.append('predict ' + columns_name[i])
            columns.append('label ' + columns_name[i])
        self.df = pd.DataFrame(columns=columns)


    def updateOutputLog(self, output, yb, name_b):
        output = output.detach().cpu().numpy()
        yb = yb.cpu().numpy()

        output = list(output)
        yb = list(yb)
        name_b = list(name_b)
        for i in range(len(output)):
            data = {'file_name':name_b[i]}
            # class 개수 1개면 문제 생겨서 나눔.
            if self.num_classes != 1:
                for j in range(self.num_classes):
                    data['predict ' + self.columns_name[j]] = output[i][j]
                    data['label ' + self.columns_name[j]] = yb[i][j]
            else:
                data['predict ' + self.columns_name[0]] = output[i]
                data['label ' + self.columns_name[0]] = yb[i]
            new_row = pd.DataFrame(data=data, index=['file_name'])
            self.df = pd.concat([self.df,new_row], ignore_index=True)

    def saveOutputLog(self, epoch, opt):

        if opt is None:
            if not os.path.exists('temp'):
                os.mkdir('temp')
            self.df.to_csv('temp/valid_output_data.csv')
            mlflow.log_artifact('temp/valid_output_data.csv', f'output_epoch_{epoch}')

        else:
            if not os.path.exists('temp'):
                os.mkdir('temp')
            self.df.to_csv('temp/train_output_data.csv')
            mlflow.log_artifact('temp/train_output_data.csv', f'output_epoch_{epoch}')