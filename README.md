## Argument
* --data_path, default='/home/work/resized_image_datas/image_5class_5000/224/', type=str  # data path
* --name, default='proto' type=str                          # Overall name for mlflow
* --epochs, default=10, type=int                            # epoch 크기
* --lr, --learning_rate, default=1e-5, type=float           # learning rate
* --mode, default='train', type=str, choices=('train','test') # train 모드, test 모드 설정
  

# pip install
```
pip install timm && pip install einops && pip install --upgrade huggingface_hub
```
```
pip install transformers datsets accelerate nvidia-ml-py3
```

# 예시 실행 코드
## vit
```
python manage.py --model_type 'vit' --model_name 'vit_base_patch16_224.augreg2_in21k_ft_in1k' --image_size 224 --epochs 10 --batch_size 16 --log_epoch 10 --data_path '/home/work/original_cropped_image_dataset/image_5class_6000/448/' 
```
```
python manage.py --model_type 'vit' --model_name 'vit_base_patch32_clip_448.laion2b_ft_in12k_in1k' --image_size 448 --epochs 30 --batch_size 16 --log_epoch 10 --data_path '/home/work/deeplant_data/' --algorithm 'regression' --columns 4 5 6 7 8 --num_classes 5 --index 9 --run_name 'dp5-vit-448-16-32'  
```
## cnn
```
python manage.py --run_name 'efficientnet-448-16' --model_type 'cnn' --model_name 'tf_efficientnetv2_l.in21k_ft_in1k' --image_size 448 --epochs 50 --batch_size 16 --log_epoch 10 --data_path '/home/work/original_cropped_image_dataset/image_5class_6000/448/' 
```
