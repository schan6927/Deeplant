import torch
import timm
import importlib

def create_model(model_cfgs):

    model_cfg = model_cfgs['models']

    module = model_cfg['module']
    model_name = model_cfg["model_name"]
    islogged = model_cfg['islogged']

    if not islogged:
        if module == 'timm':
            temp_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        else:
            temp_module = importlib.import_module(module)
            temp_model = temp_module.create_model()
    else:
        temp_model = torch.load(model_name)

    return temp_model
    