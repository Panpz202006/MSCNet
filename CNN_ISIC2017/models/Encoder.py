import torch
import torch.nn as nn

from models.vgg import VGG

class Encoder(nn.Module):
    def __init__(self,in_channels=16) -> None:
        super().__init__()
        self.encoder=VGG(in_channels)
    
    def forward(self,x):
        x_base,x1,x2,x3,x4,x5=self.encoder(x)
        return x_base,x1,x2,x3,x4,x5
        