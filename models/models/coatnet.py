import torch
import timm
import torch.nn.functional as F
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor 
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LastModule(nn.Module):
    def __init__(self):
        super().__init__()
        output_shape = 5
        input_shape = 1536
        self.fc0 = nn.Linear(input_shape, input_shape * 2, bias=True)
        self.fc1 = nn.Linear(input_shape * 2, output_shape, bias=True)
    def forward(self, x):
        x=F.gelu(self.fc0(x))
        x=self.fc1(x)
        return x


class CoatNet(nn.Module):
    def __init__(self):
        super(CoatNet, self).__init__()
        self.models=[]
        self.algorithm = 'regression'
        num_classes = 5

        temp_model = timm.create_model('coatnet_3_rw_224.sw_in12k',pretrained=True, num_classes=num_classes)
        feature={'head.global_pool' : 'out'}
        extractor = create_feature_extractor(temp_model, return_nodes = feature)
        self.models.append(extractor.to(device))
        self.lastfc_layer = LastModule()

    def forward(self, inputs):
        outputs = []
        
        
        for idx, model in enumerate(self.models):
            input = inputs[idx].to(device)
            x = model(input)
            try:
                x = x['out']
            except:
                None
            outputs.append(x)

        output = torch.concat(outputs,dim=-1)
        output = self.lastfc_layer(output)
        return output
    
    def getAlgorithm(self):
        return self.algorithm


def create_model():
    model = CoatNet()
    return model
