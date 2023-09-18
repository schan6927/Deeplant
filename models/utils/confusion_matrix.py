import matplotlib.pyplot as plt
import mlflow
import seaborn as sns

       
def save_plot(con_mat, fold, epoch):    
    cfs=sns.heatmap(con_mat,annot=True)
    cfs.set(title='Confusion Matrix', ylabel='True lable', xlabel='Predict label')
    figure = cfs.get_figure()
    
    # Save the plot as an image
    mlflow.log_figure(figure, "Confusion_Matrix_"+str(fold)+'_'+str(epoch)+'.jpg')
    
    #close figure
    plt.clf()
