import matplotlib.pyplot as plt
import mlflow
import seaborn as sns

def datasetHistogram(train_df, val_df, grade, name):
    fig = plt.figure(figsize=(30,20))
    i=0
    for g in grade:
        for n in name:
            i += 1
            plt.subplot(4,5,i)
            g_val_df = val_df[val_df['Rank'] == g]
            g_train_df = train_df[train_df['Rank'] == g]
            plt.hist(g_val_df[n], bins=10, width=0.25, alpha=0.5)
            plt.hist(g_train_df[n], bins=10, width=0.25, alpha=0.5)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of {g} {n}')
            plt.legend(labels=['valid', 'train'])
    mlflow.log_figure(fig, "Dataset_Histogram.jpg")
    plt.clf()
    
def datasetKDE(train_df, val_df, grade, name):
    fig = plt.figure(figsize=(30,20))
    i=0
    for g in grade:
        for n in name:
            i+=1
            plt.subplot(4,5,i)
            g_val_df = val_df[val_df['Rank'] == g]
            g_train_df = train_df[train_df['Rank'] == g]
            sns.kdeplot(g_val_df[n], bw='0.1')
            sns.kdeplot(g_train_df[n], bw='0.1')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.title(f'Density Plot of {g} {n}')
            plt.legend(labels=['valid', 'train'])
    mlflow.log_figure(fig, "Dataset_KDE.jpg")
    plt.clf()

def outputKDE(df, name, epoch):
    df['grade'] = df['file_name'].str.split('_').str[3]
    grade = ['1++', '1+', '2', '3']
    fig = plt.figure(figsize=(30,5))
    i=0
    for n in name:
      i += 1
      plt.subplot(1,5,i)
      sns.kdeplot(df[f'label {n}'],bw=0.1)
      sns.kdeplot(df[f'predict {n}'],bw=0.1)
      plt.xlabel('Value')
      plt.ylabel('Density')
      plt.title(f'Density Plot of {n}')
      plt.legend(labels=['label', 'predict'])
    mlflow.log_figure(fig, f'output_epoch_{epoch}/output(all)_KDE.jpg')
    plt.clf()

    fig = plt.figure(figsize=(30,20))
    i=0
    for g in grade:
      for n in name:
        i+=1
        plt.subplot(4,5,i)
        g_df = df[df['grade'] == g]
        sns.kdeplot(g_df[f'label {n}'],bw=0.1)
        sns.kdeplot(g_df[f'predict {n}'],bw=0.1)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title(f'Density Plot of {g} {n}')
        plt.legend(labels=['label', 'predict'])
    mlflow.log_figure(fig, f'output_epoch_{epoch}/output(grade)_KDE.jpg')
    plt.clf()
