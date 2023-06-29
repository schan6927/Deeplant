import os
import yaml
import pandas as pd

mlruns_path = "../../mlruns"
train_label_path = '/home/work/resized_image_datas/image_5class_5000/train_all.csv'
valid_label_path = '/home/work/resized_image_datas/image_5class_5000/valid_all.csv'
train_df = pd.read_csv(train_label_path)
valid_df = pd.read_csv(valid_label_path)
label_df = pd.concat([train_df,valid_df], ignore_index=True)

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f'YAML 파일을 읽는 중 오류가 발생했습니다: {e}')

# mlruns에 있는 모든 run을 돌면서 incorrect_data를 기존 라벨과 merge를 함
def convert_all():
    for experiment in os.listdir(mlruns_path):
        if not experiment.endswith('.yaml'):
            experiment_path = os.path.join(mlruns_path, experiment)
            convert_experiment(experiment_path)

# expriment안의 모든 runs
def convert_experiment(experiment_path):
    for run in os.listdir(experiment_path):
        if not run.endswith('.yaml'):
            run_path = os.path.join(experiment_path, run)
            convert_run(run_path)

# 하나의 run
def convert_run(run_path):
    yaml_path = os.path.join(run_path, "meta.yaml")
    data = read_yaml_file(yaml_path)
    experiment_id = data['experiment_id']
    run_id = data['run_id']

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

    # 기본 저장 장소
    save_path = experiment_id+'/'+run_id+'.csv'
    merge_df.to_csv(save_path)

# 아래처럼 원하는 함수 직접 선언하고 파일 실행하면 됨
# convert_experiment("../../mlruns/2/")