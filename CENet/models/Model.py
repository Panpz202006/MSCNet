import torch
import torch.nn as nn

from micro import CNN
from models.Decoder import Decoder, Final_Projection
from models.Encoder import Encoder

class Model(nn.Module):
    def __init__(self,in_channels=[64,128,256,512,512],scale_factor=[1,2,4,8,16],model=CNN) -> None:
        super().__init__()
        self.encoder=Encoder(in_channels,model=model)
        self.decoder=Decoder(in_channels)
        self.final=Final_Projection(in_channels,scale_factor)

    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        x=self.final(x)
        return x