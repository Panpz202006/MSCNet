import torch
import torch.nn as nn

from models.Decoder import Decoder, Final_Projection
from models.Encoder import Encoder




class Model(nn.Module):
    def __init__(self,in_channels=3,embed_channels=64) -> None:
        super().__init__()
        self.encoder=Encoder(embed_channels)
        # self.fusion=Fusion(embed_channels)
        self.decoder=Decoder(embed_channels)
        self.final=Final_Projection(embed_channels)

    def forward(self,x):
        x1,x2,x3,x4=self.encoder(x)
        # x1,x2,x3,x4,x5=self.fusion(x1,x2,x3,x4,x5)
        x1,x2,x3,x4=self.decoder(x1,x2,x3,x4)
        x1,x2,x3,x4=self.final(x1,x2,x3,x4)
        return x1,x2,x3,x4
        
