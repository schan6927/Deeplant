
import timm
pretrained_model_list = timm.list_models('*',pretrained=True)
for model in pretrained_model_list:
    print(model)