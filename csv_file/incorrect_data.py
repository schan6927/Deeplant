import os
import yaml
import pandas as pd

mlruns_path = "../mlruns"
data_label_path = '/home/work/image....'
label_df = pd.read_csv(data_label_path)

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f'YAML 파일을 읽는 중 오류가 발생했습니다: {e}')

def convert_all():
    for experiment in os.listdir(mlruns_path):
        if not experiment.endswith('.yaml'):
            experiment_path = os.path.join(mlruns_path, experiment)
            convert_experiment(experiment_path)
        

def convert_experiment(experiment_path):
    for run in os.listdir(experiment_path):
        if not run.endswith('.yaml'):
            run_path = os.path.join(experiment_path, run)
            convert_run(run_path)

def convert_run(run_path):
    yaml_path = os.path.join(run_path, "meta.yaml")
    data = read_yaml_file(yaml_path)
    experiment_id = data['experiment_id']
    run_name = data['run_name']

    artifact_path = os.path.join(run_path,"artifacts")
    csv_path = os.path.join(artifact_path,"incorrect_data.csv")
    if not os.path.exists(csv_path):
      print("Can not find csv")
      return
    
    incorrect_df = pd.read_csv(csv_path)
    merge_df = pd.merge(incorrect_df, label_df, on='file_name', how='inner', sort=True)
    if not os.path.exists(experiment_id):
        os.mkdir(experiment_id)
    merge_df.reset_index(drop=True)

    save_path = experiment_id+'/'+run_name+'.csv'
    merge_df.to_csv(save_path)

