import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from timm.models.layers import DropPath
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
#from selective_scan import selective_scan_fn as selective_scan_fn_v1

class S6Block(nn.Module):
    def __init__(self,in_channel,k=4,state_rank=16) -> None:
        super().__init__()
        self.in_channel=in_channel
        self.delta_rank=int(in_channel/state_rank)   #
        self.state_rank=state_rank
        #将特征图x映射成delta、B、C
        self.x_proj_weight=nn.Linear(in_channel, self.delta_rank + state_rank * 2, bias=False)
        #(k,d,c)
        #(192/16+16*2, 192)*4
        #(384/16+16*2, 384)*4
        #(768/16+16*2, 768)*4
        self.x_proj_weights = nn.Parameter(torch.stack([self.x_proj_weight.weight for i in range(k)], dim=0))
        del self.x_proj_weight

        #得到delta
        self.delta_projs_weight=self.delta_init(self.delta_rank, in_channel)
        #(k,c,r)
        self.delta_projs_weights = nn.Parameter(torch.stack([self.delta_projs_weight.weight for i in range(k)], dim=0))
        self.delta_projs_bias = nn.Parameter(torch.stack([self.delta_projs_weight.bias for i in range(k)], dim=0))
        del self.delta_projs_weight

        #得到Ds
        self.Ds = self.D_init(in_channel, k=4)
        #得到As
        self.As = self.A_init(state_rank, in_channel, k=4)

    @staticmethod
    def D_init(in_channel, k=1, device=None, merge=True):
        D = torch.ones(in_channel, device=device)
        if k > 1:
            D = repeat(D, "c1 -> r c1", r=k)
            if merge:
                D = D.flatten(0, 1)
        #(k*c,1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D
    
    @staticmethod
    def delta_init(delta_rank, in_channel, delta_scale=1.0, delta_min=0.001, delta_max=0.1, delta_init_floor=1e-4):
        delta_proj = nn.Linear(delta_rank, in_channel, bias=True)
        # Initialize special dt projection to preserve variance at initialization
        delta_init_std = delta_rank**-0.5 * delta_scale
        nn.init.uniform_(delta_proj.weight, -delta_init_std, delta_init_std)
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        delta = torch.exp(
            torch.rand(in_channel) * (math.log(delta_max) - math.log(delta_min))
            + math.log(delta_min)
        ).clamp(min=delta_init_floor)
        #Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inverse_delta = delta + torch.log(-torch.expm1(-delta))
        with torch.no_grad():
            delta_proj.bias.copy_(inverse_delta)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        delta_proj.bias._no_reinit = True
        return delta_proj
    
    @staticmethod
    def A_init(state_rank, in_channel, k=1, device=None, merge=True):
        #(c,s)
        A = repeat(
            torch.arange(1, state_rank + 1, dtype=torch.float32, device=device),
            "s -> c s",
            c=in_channel,
        ).contiguous()
        
        A_log = torch.log(A)
        if k > 1:
            A_log = repeat(A_log, "c s -> r c s", r=k)
            if merge:
                A_log = A_log.flatten(0, 1)
        #(k*c,s)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log
    
    def forward(self,x:torch.Tensor):
        '''
        x (torch.Tensor): 输入 维度：(B, H, W, C)
        '''
        B, C, H, W = x.shape
        L = H * W
        K = 4
        #用于从四个方向扫描特征图x
        x=self.get_x(x)
        deltas,Bs,Cs,delta_projs_bias=self.get_DBC(x,B,C,H,W)
        As,Ds=self.get_AD()

        x=x.float().view(B,-1,L)
        y=selective_scan_fn(x,deltas,As,Bs,Cs,Ds,z=None,delta_bias=delta_projs_bias,delta_softplus=True,return_last_state=False)
        y=y.view(B,K,-1,L)
        assert y.dtype == torch.float
        y2 = torch.flip(y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        y3 = torch.transpose(y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y4 = torch.transpose(y2[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y[:, 0], y2[:, 0], y3, y4


    def get_x(self,x):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        x = torch.cat([x, torch.flip(x, dims=[-1])], dim=1) # (b, k, c, l)
        return x

    def get_DBC(self,x,B,C,H,W):
        L = H * W
        K = 4
        x_dbc = torch.einsum("b k c l, k d c -> b k d l", x.view(B, K, -1, L), self.x_proj_weights)
        #Bs (b,k,s,l)   Cs (b,k,s,l)
        deltas, Bs, Cs = torch.split(x_dbc, [self.delta_rank, self.state_rank, self.state_rank], dim=2)
        #deltas (b,k,c,l)
        deltas = torch.einsum("b k r l, k c r -> b k c l", deltas.view(B, K, -1, L), self.delta_projs_weights)
        #deltas (b,k*c,l)
        deltas = deltas.contiguous().float().view(B, -1, L)
        #Bs (b,k,s,l)
        Bs = Bs.float().view(B, K, -1, L)
        #Cs (b,k,s,l)
        Cs = Cs.float().view(B, K, -1, L)
        #(k*c)
        delta_projs_bias = self.delta_projs_bias.float().view(-1)
        return deltas,Bs,Cs,delta_projs_bias

    def get_AD(self):
        #(k*c)
        Ds = self.Ds.float().view(-1)
        #(k*c,s)
        As = -torch.exp(self.As.float()).view(-1, self.state_rank)
        return As,Ds


class SS2D(nn.Module):
    def __init__(self,in_channel) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channel, in_channel * 4, bias=False)
        #分辨率不变，通道数变为两倍
        self.conv2d = nn.Conv2d(
            in_channels=in_channel*2,
            out_channels=in_channel*2,
            groups=in_channel*2,
            bias=True,
            kernel_size=3,
            padding=(3 - 1) // 2
        )
        self.act = nn.SiLU()
        self.s6block=S6Block(in_channel=in_channel*2)
        self.norm = nn.LayerNorm(in_channel*2)
        self.act2=nn.SiLU()
        self.fc2=nn.Linear(in_channel*2, in_channel, bias=False)
        
    
    def forward(self,x:torch.Tensor):
        '''
        x (torch.Tensor): 输入 维度：(B, H, W, C)
        输出维度:  (B,H,W,C)
        '''
        B,H,W,C=x.shape
        xx = self.fc1(x)
        x_ssm,x=xx.chunk(2, dim=-1)
        #(B,C,H,W)
        x_ssm = x_ssm.permute(0, 3, 1, 2).contiguous()
        x_ssm=self.conv2d(x_ssm)
        x_ssm = self.act(x_ssm)
        y1, y2, y3, y4 = self.s6block(x_ssm)
        assert y1.dtype == torch.float32
        #将四个方向的特征信息，都加起来
        y = y1 + y2 + y3 + y4
        #（B,H,W,C)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y=self.norm(y)
        y=y*self.act2(x)
        y = self.fc2(y)
        return y


class VSSBlock(nn.Module):
    def __init__(self,in_channel,drop_prob):
        super().__init__()
        self.norm = nn.LayerNorm(in_channel)
        self.ss2d=SS2D(in_channel)
        self.drop_path = DropPath(drop_prob)

        self.norm2=nn.LayerNorm(in_channel)
        self.fc1=nn.Linear(in_channel, in_channel*4)
        self.conv=nn.Conv2d(in_channel*4, in_channel*4, 3, 1, 1, bias=True, groups=in_channel*4)
        self.act=nn.GELU()
        self.fc2=nn.Linear(in_channel*4, in_channel)

    def forward(self,x:torch.Tensor):
        '''
        x (torch.Tensor): 输入 维度：(B, H, W, C)
        输出 维度：(B, H, W, C)
        '''
        x=x.permute(0,2,3,1).contiguous()
        short_cut=x
        x=self.norm(x)
        x=self.ss2d(x)
        x=short_cut+self.drop_path(x)
        short_cut=x
        x=self.norm2(x)
        x=self.fc1(x)
        x=x.permute(0,3,1,2).contiguous()
        x=self.conv(x)
        x=self.act(x)
        x=x.permute(0,2,3,1).contiguous()
        x=self.fc2(x)
        x=short_cut+x
        x=x.permute(0,3,1,2).contiguous()
        return x

