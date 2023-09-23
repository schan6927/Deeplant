from torch import nn
import torch
import numpy as np

class Accuracy():
    def __init__(self, length, num_classes, mode):
        
        self.num_classes = num_classes
        self.length = length
        self.mode = mode

        self.cumulative_metric = np.zeros(num_classes)

    def update(self, output, yb):
        if self.mode == 'classification':
            _, pred_b = torch.max(output.data,1)
            metric_b = (pred_b == yb).sum().item()
            self.cumulative_metric += metric_b

        elif self.mode == 'regression':
            if self.num_classes != 1:
                for i in range(self.num_classes):
                    self.cumulative_metric[i] += (torch.round(output[:,i]) == yb[:,i]).sum().item()
            else:
                self.cumulative_metric += (torch.round(output) == yb).sum().item()
        
    def getResult(self):
        return self.cumulative_metric / self.length
    
    def getClassName():
        return "Accuracy"
    

class R2score():
    def __init__(self, length):
        self.length = length
        self.cumulative_y = None
        self.cumulative_output = None
    
    def update(self, output, yb):
        output = output.detach().cpu().numpy()
        yb = yb.cpu().numpy()
        if self.cumulative_y is None:
            self.cumulative_y = np.array(yb)
        else:
            self.cumulative_y = np.vstack((self.cumulative_y,yb))
            
        if self.cumulative_output is None:
            self.cumulative_output = np.array(output)
        else:
            self.cumulative_output = np.vstack((self.cumulative_output,output))

    def getResult(self):
        y_mean = self.cumulative_y.mean(axis=0)
        ssr = np.square(self.cumulative_y - self.cumulative_output).sum(axis=0)
        sst = np.square(self.cumulative_y - y_mean).sum(axis=0)
        r2_score = (1 - (ssr / sst)).mean()
        return r2_score
    
    def getClassName():
        return "R2score"
    


class MeanAbsError():
    def __init__(self, length, num_classes):
        self.num_classes = num_classes
        self.cumulative_metric = np.zeros(num_classes)
        self.length = length
    
    def update(self, output, yb):
        if self.num_classes != 1:
            for i in range(self.num_classes):
                self.cumulative_metric[i] += torch.abs(output[:,i] - yb[:,i]).sum().item()
        else:
            self.cumulative_metric += torch.abs(output - yb).sum().item()

    def getResult(self):
        return self.cumulative_metric / self.length
    
    def getClassName():
        return "MAE"




def initMetrics(eval_functions):
    metrics = []
    for f in eval_functions:
        if f == 'ACCURACY':
            metrics.append(Accuracy())

