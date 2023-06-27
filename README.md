# Deeplant

## Argument
* --data_path, default='/home/work/resized_image_datas/image_5class_5000/224/', type=str  # data path

* --model_type, type=str, choices('cnn','vit')              # 사용할 모델 선택
* --model_name, type=str                                    # 사용할 모델 선택
* --sanity, default=False, type=bool                        # 빠른 test 여부

* --image_size, default=224, type=int, choices=(224,448)    # 이미지 크기 재설정
* --num_workers, default=4, type=int                        # 훈련에 사용할 CPU 코어 수

* --epochs, default=10, type=int                            # fold당 epoch
* --kfold, default=5, type=int                              # kfold 사이즈
* --batch_size, default=16, type=int                        # 배치 사이즈
* --lr, --learning_rate, default=1e-5, type=float           # learning rate
* --log_epoch, default=5, type=int                          # 몇 epoch당 모델을 기록할 지 정함
* --num_classes, default=5, type=int                        # output class 개수

* --factor, default=0.5, type=float                         # scheduler factor
* --threshold, default=0.003, type=float                    # scheduler threshold
* --momentum', default=0.9, type=float                      # optimizer의 momentum
* --weight_decay, --wd, default=5e-4, type=float            # 가중치 정규화

* --pretrained, default=True, type=bool                     # pre-train 모델 사용 여부
* --load_run, default=False, type=bool                      # run의 모델 사용 여부
* --logged_model, default=None, type=str                    # 사용할 run의 path

> 추가 예정
>> * optimizer
>> * patience

# pip install
pip install timm && pip install einops && pip install --upgrade huggingface_hub

# 기본 vit 실행 코드
python manage.py --model_type 'vit' --model_name 'vit_base_patch16_224.augreg2_in21k_ft_in1k' --image_size 224 \
--epochs 10 --batch_size 16 --log_epoch 5 