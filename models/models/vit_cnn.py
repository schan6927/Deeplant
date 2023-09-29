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

class ViTCnnModel(nn.Module):
    def __init__(self):
        super(ViTCnnModel,self).__init__()
        self.algorithm = "regression"
        self.fc_input_shape = 0
        model_1 = timm.create_model("vit_base_patch32_clip_448.laion2b_ft_in12k_in1k", pretrained=True, num_classes=5)
        model_2 = timm.create_model("resnetrs152.tf_in1k", pretrained=True, num_classes=5)
        self.model_1 = create_feature_extractor(model_1, return_nodes={"fc_norm":"out"})
        self.model_2 = create_feature_extractor(model_2, return_nodes={"global_pool.flatten":"out"})
        
        self.fc_input_shape += self.model_1.state_dict()[list(self.model_1.state_dict())[-1]].shape[-1]
        self.fc_input_shape += self.model_2.state_dict()[list(self.model_2.state_dict())[-1]].shape[-1]
        self.fc_layer = LastModule(self.fc_input_shape)
    
    
    def forward(self, inputs):
        input_1 = inputs[0].to(device)
        input_2 = inputs[1].to(device)
        
        output_1 = self.model_1(input_1)['out']
        output_2 = self.model_2(input_2)['out']
        
        output = torch.concat([output_1, output_2],dim=-1)
        output = self.fc_layer(output)
        return output

    def getAlgorithm(self):
        return self.algorithm


def create_model():
    model = ViTCnnModel()
    return model
