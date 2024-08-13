import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super().__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1=ConvNormAct(in_channels,in_channels,1,norm=None,act=nn.ReLU)
        self.conv2=ConvNormAct(in_channels,in_channels,1,norm=None,act=nn.Sigmoid)

    def forward(self, x):
        x_sig=self.conv2(self.conv1(x))
        return x_sig*x+x



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
            nn.Sigmoid()
        )
    

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        att = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(att)
        return out*x+x


class EdgeEnhance(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.AP=nn.AvgPool2d(3,1,1)
        # self.sig=nn.Sigmoid()
        self.conv=ConvNormAct(in_channels,in_channels,1,act=nn.Sigmoid)
        
    def forward(self,x):
        x_ap=self.conv(x-self.AP(x))
        x_ee=x_ap*x+x
        return x_ee


class ConvNormAct(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 norm=nn.BatchNorm2d,
                 act=nn.ReLU):
        super().__init__()
        self.conv=nn.Conv2d(in_channels, out_channels, kernel_size,stride,padding,dilation,groups)
        self.in_channels=in_channels
        self.out_channels=out_channels

        if norm!=None:
            self.norm=norm(out_channels)
        else:
            self.norm=None
        if act!=None:
            self.act=act()
        else:
            self.act=None
        
    def forward(self,x):
        x=self.conv(x)
        if self.norm!=None:
            x=self.norm(x)
        if self.act!=None:
            x=self.act(x)
        return x