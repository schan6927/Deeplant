import torch
import timm
import torch.nn.functional as F
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor
import importlib
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LastModule(nn.Module):
    def __init__(self,params):
        super().__init__()
        input_shape = params['input_shape']
        output_shape = params['output_shape']
        self.fc1 = nn.Linear(input_shape, input_shape*2, bias=True)
        self.fc2 = nn.Linear(input_shape*2, output_shape, bias=True)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

class Model(nn.Module):
    def __init__(self,model_cfgs):
        super(Model,self).__init__()
        
        self.fc_input_shape = 0
        self.models = []

        print('------Making model------')
        for model_cfg in model_cfgs['models']:
            module = model_cfg['module']
            model_name = model_cfg['model_name']
            islogged = model_cfg['islogged']
            features = model_cfg['features']
            pretrained = model_cfg['pretrained']
            num_classes = model_cfg['num_classes']
            
            print(model_cfg)
            
            if not islogged:
                if module == 'timm':
                    temp_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
                else:
                    temp_module = importlib.import_module(module)
                    temp_model = temp_module.create_model(model_name)
            else:
                temp_model = torch.load(model_name)

            if features: 
                feature_extractor = create_feature_extractor(temp_model,return_nodes=features)
                temp_model = feature_extractor
            temp_model = temp_model.to(device)
            self.fc_input_shape += temp_model.state_dict()[list(temp_model.state_dict())[-1]].shape[-1]
            self.models.append(temp_model)
            
            print('finish')
            
        print('Check fc_layer')
        if model_cfgs['fc_layer']:
            print('Make fc_layer ...')
            model_cfgs['fc_layer']['input_shape'] = self.fc_input_shape
            self.lastfc_layer = LastModule(model_cfgs['fc_layer'])
            print('finish')
        else:
            print('fc_layer is None')
            self.lastfc_layer = None
        print('-----End-----')

        #interoutput
        self.interoutput = None

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
        if self.lastfc_layer:
            output = self.lastfc_layer(output)
            
        return output
    
    def shap_layer(self):
        return self.interoutput
