import matplotlib.pyplot as plt
import mlflow
import seaborn as sns

class SaveCM:
    def __init__(self,con_mat,epoch):
        self.con_mat=con_mat
        self.epoch=epoch
        
    def save_plot(self,fold):    
        cfs=sns.heatmap(self.con_mat,annot=True)
        cfs.set(title='Confusion Matrix', ylabel='True lable', xlabel='Predict label')
        figure = cfs.get_figure()
        
        # Save the plot as an image
        mlflow.log_figure(figure, f"Confusion_Matrix_{fold}.png")