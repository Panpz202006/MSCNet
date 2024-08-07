import torch
import torch.nn as nn

from models.backbone.vgg import VGG as VGG_backbone
from micro import DAE_FORMER, ULTRALIGHT_VM_UNET, VGG,VMUNET
from models.backbone import DAEFormer,UltraLight_VM_UNet, vmamba

class Encoder(nn.Module):
    def __init__(self,in_channels=[64,128,256,512,512],model=VGG) -> None:
        super().__init__()
        if model==VGG:
            self.encoder=VGG_backbone(in_channels)
        elif model==VMUNET:
            self.encoder=vmamba.VSSM()
        elif model==ULTRALIGHT_VM_UNET:
            self.encoder=UltraLight_VM_UNet.UltraLight_VM_UNet()
        elif model==DAE_FORMER:
            self.encoder=DAEFormer.DAEFormer()
    def forward(self,x):
        x=self.encoder(x)
        return x
