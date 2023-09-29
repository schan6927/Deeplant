import torch
import torch.nn.functional as F
from torch import nn
import timm
from torchvision.models.feature_extraction import create_feature_extractor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LastModule(nn.Module):
    def __init__(self):
        super().__init__()
        input_shape = 798
        output_shape = 5
        self.fc1 = nn.Linear(input_shape, input_shape*2, bias=True)
        self.fc2 = nn.Linear(input_shape*2, output_shape, bias=True)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

class num_model(nn.Module):
    def __init__(self):
        super().__init__()
        input_shape = 4
        output_shape = 30
        self.fc1 = nn.Linear(input_shape, input_shape*2, bias=True)
        self.fc2 = nn.Linear(input_shape*2, output_shape, bias=True)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class test_model(nn.Module):
    def __init__(self):
        super(test_model,self).__init__()
        self.algorithm = "regression"
        
        # 첫 층의 모델들을 저장함.
        # 모델 위에 모델을 올릴거면 잘 고쳐야함. 
        self.models=[]

        temp_model = timm.create_model("vit_base_patch32_clip_448.laion2b_ft_in12k_in1k", pretrained=True, num_classes=5)
        features={'fc_norm':'out'} # fc_norm, global_pool.flatten 
        feature_extractor = create_feature_extractor(temp_model, return_nodes = features)
        numeric_model = num_model()

        self.models.append(feature_extractor.to(device))
        self.models.append(numeric_model.to(device))

        self.lastfc_layer =  LastModule()


    def forward(self, inputs):
        outputs = []
        
        #-------------첫 층의 결과를 계산함----------
        for idx, model in enumerate(self.models):
            input = inputs[idx].to(device)
            x = model(input)
            try:
                x = x['out']
            except:
                None
            outputs.append(x)
        #------------------------------------------
        output = torch.concat(outputs,dim=-1) # 결과들을 concat함
    
        output = self.lastfc_layer(output) # 다음 레이어 계산
            
        return output

    def getAlgorithm(self):
        return self.algorithm

def create_model():
    model = test_model()
    return model
