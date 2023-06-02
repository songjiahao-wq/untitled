import torch
import torch.nn as nn
x = torch.randn(1,3,640,640)
print(nn.AdaptiveAvgPool2d(1)(x).shape)