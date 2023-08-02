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
        num_classes=params['num_classes']
        pretrained=params['pretrained']
        model_name=params['model_name']
        logged_model=params['logged_model']
        self.custom_fc=params['custom_fc']
        #--------------------vit-----------------------
        if logged_model is None:
            temp_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, exportable=True )
        else:
            temp_model = torch.load(logged_model)

        if self.custom_fc == True:
            features={'fc_norm':'out'} # fc_norm, global_pool.flatten 
            feature_extractor = create_feature_extractor(temp_model,return_nodes = features)
            vmodel = feature_extractor
            self.lastfc_layer =  LastModule(params)
        else:
            vmodel = temp_model

        self.vision_model = vmodel
        self.numeric_model = None

        #interoutput
        self.interoutput = None

    def forward(self, image, num=None):
        
        if self.custom_fc == True:
            image_output = self.vision_model(image)['out'] # [batch, 768]
        else:
            image_output = self.vision_model(image)

        if num is not None:
            if self.numeric_model is None:
                num_output = num
            else:
                num_output = self.numeric_model(num)
            output = torch.cat([image_output, num_output], dim=-1)
        else:
            output = image_output

        self.interoutput = output
        if self.custom_fc == True:
            output = self.lastfc_layer(output)
            
        return output
    
    def shap_layer(self):
        return self.interoutput

