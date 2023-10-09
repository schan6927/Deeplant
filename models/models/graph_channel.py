import torch
import torch.nn.functional as F
from torch import nn
import timm
from torchvision.models.feature_extraction import create_feature_extractor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LastModule(nn.Module):
    def __init__(self, input_shape):
        super().__init__() 
        output_shape = 5
        self.fc1 = nn.Linear(input_shape, input_shape*2, bias=True)
        self.fc2 = nn.Linear(input_shape*2, output_shape, bias=True)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class GCModel(nn.Module):
    def __init__(self):
        super(GCModel,self).__init__()
        self.algorithm = "regression"
        self.fc_input_shape = 0
        model_1 = timm.create_model("vit_base_patch32_clip_448.laion2b_ft_in12k_in1k", pretrained=True, num_classes=5, in_chans=4)
        self.model_1 = create_feature_extractor(model_1, return_nodes={"fc_norm":"out"})
        
        self.fc_input_shape += self.model_1.state_dict()[list(self.model_1.state_dict())[-1]].shape[-1]
        self.fc_layer = LastModule(self.fc_input_shape)
    
    
    def forward(self, inputs):
        x = None
        for input in inputs:
            input = input.to(device)
            if x is None:
                x = input
            else:
                x = torch.concat([x,input],dim=1)
            
        output = self.model_1(x)['out']
        output = self.fc_layer(output)
        return output

    def getAlgorithm(self):
        return self.algorithm


def create_model():
    model = GCModel()
    return model
