import torch
import torch.nn as nn

from models import vmamba
from models.vgg import VGG

class Encoder(nn.Module):
    def __init__(self,in_channels=16) -> None:
        super().__init__()
        self.encoder=vmamba.VSSM()
        self.up=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self,x:torch.Tensor):
        x1,x2,x3,x4=self.encoder(x)
        x1=self.up(x1.permute(0,3,1,2).contiguous())
        x2=self.up(x2.permute(0,3,1,2).contiguous())
        x3=self.up(x3.permute(0,3,1,2).contiguous())
        x4=x4.permute(0,3,1,2).contiguous()
        return x1,x2,x3,x4
        