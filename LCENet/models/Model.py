import torch
import torch.nn as nn

from micro import VGG
from models.Decoder import Decoder,Final_Supervise
from models.Encoder import Encoder

class Model(nn.Module):
    def __init__(self,in_channels=[64,128,256,512,512],scale_factor=[1,2,4,8,16],model=VGG) -> None:
        super().__init__()
        self.encoder=Encoder(in_channels,model=model)
        self.decoder=Decoder(in_channels)
        self.final=Final_Supervise(in_channels,scale_factor)

    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        x=self.final(x)
        return x