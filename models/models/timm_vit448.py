import torch
import torch.nn.functional as F
from torch import nn
import timm
from torchvision.models.feature_extraction import create_feature_extractor
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TimmModel(nn.Module):
    def __init__(self):
        super(TimmModel,self).__init__()
        self.algorithm = "classification"
        self.model = timm.create_model("vit_base_patch32_clip_448.laion2b_ft_in12k_in1k", pretrained=True, num_classes=5)
    def forward(self, inputs):
        input = inputs[0].to(device)
        output = self.model(input)
        
        return output

    def getAlgorithm(self):
        return self.algorithm

def create_model():
    model = TimmModel()
    return model
