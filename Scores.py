import pandas as pd
import numpy as np
import os

class Scores:
    def __init__(self, epoch):
        self.epoch = epoch

    def wrongAns(self, output, name_b, num_classes, yb, columns_name):
        df = pd.DataFrame()
        for i in range(len(output)):
            data = {'file_name': name_b[i]}
            if num_classes == 1:
                output[i] = [output[i]]
                yb[i] = [yb[i]]
            for j in range(num_classes):
                data['predict ' + columns_name[j]] = output[i][j]
                data['label ' + columns_name[j]] = yb[i][j]

            new_row = pd.DataFrame(data=data, index=[0])
            df = pd.concat([df, new_row], ignore_index=True)
        self.save(df)
    
    def save(self, df):
        if not os.path.exists('temp'):
            os.mkdir('temp')
        df.to_csv(f'temp/output_{self.epoch}.csv', index=False)

    def cal_R2(self, running_y, running_output, yb, output):
        if running_y is None:
            running_y = np.array(yb)
        else:
            running_y = np.vstack((running_y, yb))
        
        if running_output is None:
            running_output = np.array(output)
        else:
            running_output = np.vstack((running_output, output))

        n = len(running_y)
        y_mean = running_y.mean(axis=0)
        ssr = np.square(running_y - running_output).sum(axis=0)
        sst = np.square(running_y - y_mean).sum(axis=0)
        
        r2_score = (1 - (ssr / sst)).mean()
        print(r2_score)
        return running_y, running_output, r2_score