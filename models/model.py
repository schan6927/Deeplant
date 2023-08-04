import torch
import timm
import torch.nn.functional as F
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor

class LastModule(nn.Module):
    def __init__(self,params):
        super().__init__()
        input_shape = params['input_shape']
        output_shape = params['output_shape']
        self.fc1 = nn.Linear(input_shape, input_shape*2, bias=True)
        self.fc2 = nn.Linear(input_shape * 2, output_shape, bias=True)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

class Model(nn.Module):
    def __init__(self,params):
        super(Model,self).__init__()
        model_cfgs=params['model_cfgs']
        self.fc_input_shape = 0

        self.models = []
        for model_cfg in model_cfgs:
            model_name = model_cfg['model_name']
            islogged = model_cfg['islogged']
            features = model_cfg['features']
            pretrained = model_cfg['pretrained']
            num_classes = model_cfg['num_classes']

            #--------------------vit-----------------------
            if not islogged:
                temp_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
            else:
                temp_model = torch.load(model_name)

            if features: 
                feature_extractor = create_feature_extractor(temp_model,return_nodes = features)
                temp_model = feature_extractor
                
            self.fc_input_shape += temp_model.state_dict()[list(temp_model.state_dict())[-1]].shape[-1]
            self.models.append[temp_model]


        params['input_shape']=self.fc_input_shape
        self.lastfc_layer = LastModule(params)

        #interoutput
        self.interoutput = None

    def forward(self, inputs):
        output = None
        
        for idx, model in enumerate(self.models):
            input = inputs[idx].cuda()
            try:
                x = model(input)['out']
            except:
                x = model(input)

            if output is None:
                output = x
            else:
                output = torch.cat([output,x], dim=-1)

        self.interoutput = output
        output = self.lastfc_layer(output)
            
        return output
    
    def shap_layer(self):
        return self.interoutput

