import torch
import torch.nn as nn

from micro import CNN, MAMBA, TRANSFORMER
from models import vmamba
from models.vgg import VGG

class Encoder(nn.Module):
    def __init__(self,in_channels=[64,128,256,512,512],model=CNN) -> None:
        super().__init__()
        if model==CNN:
            self.encoder=VGG(in_channels)
        elif model==MAMBA:
            self.encoder=vmamba.VSSM()
        elif model==TRANSFORMER:
            self.encoder=None
    def forward(self,x):
        x=self.encoder(x)
        return x
        