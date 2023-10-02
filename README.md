# Deeplant
딥러닝을 이용한 육류 이미지 분석.

## Argument
* parser.add_argument('--run', default ='proto', type=str)  # run 이름 설정
* parser.add_argument('--name', default ='proto', type=str)  # experiment 이름 설정
* parser.add_argument('--model_cfgs', default='configs/model_cfgs.json', type=str)  # model 관련 config 파일 경로 설정
* parser.add_argument('--mode', default='train', type=str, choices=('train', 'test')) # 학습모드 / 평가모드. (현재 train만 가능.)
* parser.add_argument('--epochs', default=10, type=int)  #epochs
* parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float)  # learning rate
* parser.add_argument('--data_path', default='/home/work/deeplant_data', type=str)  # data path

# pip install
```
pip install timm && pip install einops && pip install --upgrade huggingface_hub
```
```
pip install transformers datsets accelerate nvidia-ml-py3
```

# 예시 실행 코드
```
python manage.py --model_type 'vit'
```
