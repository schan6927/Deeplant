from torch import nn

class Accuracy():
    def __init__(self, length):
        self.cumulative_metric = 0.0
        self.length = length

    def update(self, output, yb):
        _, pred_b = torch.max(output.data,1)
        metric_b = (pred_b == yb).sum().item()
        self.cumulative_metric += metric_b
        
    def getResult(self):
        return self.cumulative_metric / self.length
    
    def getClassName():
        return "Accuracy"




def initMetrics(eval_functions):
    metrics = []
    for f in eval_functions:
        if f == 'ACCURACY':
            metrics.append(Accuracy())

