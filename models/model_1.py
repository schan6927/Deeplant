import torch
import torch.nn.functional as F
from torch import nn

class test_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4*2, bias=True)
        self.fc2 = nn.Linear(4*2, 4, bias=True)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_model(model_name):
    if model_name == 'test_model':
        model = test_model()
    else:
        model = None
        
    return model
