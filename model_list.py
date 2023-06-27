
import timm
pretrained_model_list = timm.list_models('*resnet*',pretrained=True)
for model in pretrained_model_list:
    print(model)