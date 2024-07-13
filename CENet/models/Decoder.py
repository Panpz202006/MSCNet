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



class CatFuse(nn.Module):
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


class Decoder(nn.Module):
    def __init__(self,in_channels=[64,128,256,512,512]):
        super().__init__()
        self.num_layer=len(in_channels)
        self.decoder=nn.ModuleList()
        for i_layer in range(self.num_layer):
            self.decoder.append(DecoderBlock(in_channels[i_layer],high_level=True))
        self.cf=nn.ModuleList()
        for i_layer in range(self.num_layer-1):
            self.cf.append(CatFuse(in_channels[i_layer+1],in_channels[i_layer]))
        
    def forward(self,x):
        x_list=[]
        input=x[-1]
        for i in range(-1, -len(self.decoder)-1, -1):
            x_d=self.decoder[i](input)
            x_list.append(x_d)
            if i!=-self.num_layer:
                input=self.cf[i](x_d,x[i-1])
        return x_list



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
    def __init__(self,in_channels=[64,128,256,512,512],scale_factor=[1,2,4,8,16]):
        super().__init__()
        self.final=nn.ModuleList()
        self.num_layer=len(in_channels)
        for i_layer in range(self.num_layer):
            self.final.append(Final_ProjectionBlock(in_channels[i_layer],scale_factor[i_layer]))

    def forward(self,x):
        x=x[::-1]
        x_list=[]
        for i in range(self.num_layer):
            x_list.append(self.final[i](x[i]))
        return x_list