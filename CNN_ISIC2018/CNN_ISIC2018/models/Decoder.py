from turtle import right
import torch
import torch.nn as nn

from models.Tool import ChannelAttention, SpatialAttention
from models.Tool import ConvNormAct, EdgeEnhance

class MSCBlock(nn.Module):
    def __init__(self,in_channels,padding,dilation,high_level=False):
        super().__init__()
        self.extract=ConvNormAct(in_channels,in_channels//8,3,padding=padding,dilation=dilation)
        self.integrate=ConvNormAct(in_channels//8,in_channels//8,1)
        self.ee=EdgeEnhance(in_channels//8)
       

    def forward(self,x):
        x=self.integrate(self.extract(x))
        x=self.ee(x)
        return x
    
class MSC(nn.Module):
    def __init__(self,in_channels,high_level=False):
        super().__init__()
        self.up=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.msc1=MSCBlock(in_channels,1,1,high_level)
        self.msc2=MSCBlock(in_channels,2,2,high_level)
        self.msc3=MSCBlock(in_channels,3,3,high_level)
        self.msc4=MSCBlock(in_channels,4,4,high_level)
        self.msc5=MSCBlock(in_channels,1,1,high_level)
        self.msc6=MSCBlock(in_channels,3,3,high_level)
        self.msc7=MSCBlock(in_channels,1,1,high_level)
        self.msc8=MSCBlock(in_channels,3,3,high_level)
        
        self.integrate=nn.Sequential(
            ConvNormAct(in_channels,in_channels,3,padding=1),
            ConvNormAct(in_channels,in_channels,1),
            EdgeEnhance(in_channels),
            ChannelAttention(in_channels),
        )

    def forward(self,x):
        x1=self.msc1(x)
        x2=self.msc2(x)
        x3=self.msc3(x)
        x4=self.msc4(x)
        x5=self.pool(self.msc5(self.up(x)))
        x6=self.pool(self.msc6(self.up(x)))
        x7=self.up(self.msc7(self.pool(x)))
        x8=self.up(self.msc8(self.pool(x)))
        out=torch.cat([x1,x2,x3,x4,x5,x6,x7,x8],dim=1)
        out=self.integrate(out)
        return out



class DecoderBlock(nn.Module):
    def __init__(self,in_channels,high_level=False):
        super().__init__()
        self.msc=MSC(in_channels,high_level)
        self.integrate=nn.Sequential(
            nn.BatchNorm2d(in_channels),
            ConvNormAct(in_channels,in_channels,3,padding=1),
            ConvNormAct(in_channels,in_channels,1),
            EdgeEnhance(in_channels),
            ChannelAttention(in_channels),
        )
        self.act=nn.ReLU()
    def forward(self,x):
        x=self.act(self.msc(x)+x)
        x=self.act(self.integrate(x)+x)
        return x



class Cat(nn.Module):
    def __init__(self,in_channels1,in_channels2):
        super().__init__()
        self.pro=nn.Sequential(
            nn.BatchNorm2d(in_channels1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvNormAct(in_channels1,in_channels2,1),
            ChannelAttention(in_channels2),
        )
        self.sig=nn.Sigmoid()
        self.act=nn.ReLU()

    def forward(self,x1,x2):
        x=self.act(self.pro(x1)*self.sig(x2)+x2)
        return x

class Cat_Base(nn.Module):
    def __init__(self,in_channels1,in_channels2):
        super().__init__()
        self.pro=nn.Sequential(
            nn.BatchNorm2d(in_channels1),
            ConvNormAct(in_channels1,in_channels2,1),
            ChannelAttention(in_channels2),
        )
        self.sig=nn.Sigmoid()
        self.act=nn.ReLU()

    def forward(self,x1,x2):
        x=self.act(self.pro(x1)*self.sig(x2)+x1)
        return x

class Decoder(nn.Module):
    def __init__(self,in_channels=16):
        super().__init__()
        self.decoder5=DecoderBlock(in_channels*8,high_level=True)
        self.decoder4=DecoderBlock(in_channels*8,high_level=True)
        self.decoder3=DecoderBlock(in_channels*4,high_level=True)
        self.decoder2=DecoderBlock(in_channels*2,high_level=True)
        self.decoder1=DecoderBlock(in_channels,high_level=True)

        self.cat4=Cat(in_channels*8,in_channels*8)
        self.cat3=Cat(in_channels*8,in_channels*4)
        self.cat2=Cat(in_channels*4,in_channels*2)
        self.cat1=Cat(in_channels*2,in_channels)
        # self.cat_base=Cat_Base(in_channels,in_channels)



    def forward(self,x_base,x1,x2,x3,x4,x5):

        x5=self.decoder5(x5)
        x4=self.cat4(x5,x4)
        x4=self.decoder4(x4)

        x3=self.cat3(x4,x3)
        x3=self.decoder3(x3)

        x2=self.cat2(x3,x2)
        x2=self.decoder2(x2)

        x1=self.cat1(x2,x1)
        x1=self.decoder1(x1)

        # x1=self.cat_base(x1,x_base)
        return x1,x2,x3,x4,x5



class Final_ProjectionBlock(nn.Module):
    def __init__(self,in_channels,scale_factor=1):
        super().__init__()
        if scale_factor>1:
            self.upsample=nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        else:
            self.upsample=None
        self.pro=nn.Sequential(
            nn.Conv2d(in_channels,1,1),
            nn.Sigmoid()
        )
    
    def forward(self,x:torch.Tensor):
        if self.upsample!=None:
            x=self.upsample(x)
        x=self.pro(x)
        return x

class Final_Projection(nn.Module):
    def __init__(self,in_channels=16):
        super().__init__()
        self.final5=Final_ProjectionBlock(in_channels*8,scale_factor=2**4)
        self.final4=Final_ProjectionBlock(in_channels*8,scale_factor=2**3)
        self.final3=Final_ProjectionBlock(in_channels*4,scale_factor=2**2)
        self.final2=Final_ProjectionBlock(in_channels*2,scale_factor=2**1)
        self.final1=Final_ProjectionBlock(in_channels,scale_factor=2**0)

    def forward(self,x1,x2,x3,x4,x5):
        x5=self.final5(x5)
        x4=self.final4(x4)
        x3=self.final3(x3)
        x2=self.final2(x2)
        x1=self.final1(x1)
        return x1,x2,x3,x4,x5