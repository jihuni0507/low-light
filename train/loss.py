# 이미지 그리드 기반 TV loss는 그대로 사용하지 못함
# 가우시안 간 위치 관계에 대한 regularization 항과 위치 관계를 따로 정의할 필요가 있다

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np

class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()
        
    def forward(self, x):
    
        return x
    
class L_second(nn.Module):
    def __init__(self):
        super(L_second, self).__init__()
        
    def forward(self, x):
        
        return x
    
# And more losses....