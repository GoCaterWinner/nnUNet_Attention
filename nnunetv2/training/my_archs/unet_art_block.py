import torch
import torch.nn as nn
from .art_block import  ART_block


class _DecoderShim:
    def __init__(self, deep_supervision=True):
        self.deep_supervision = deep_supervision

class UNetARTBlock(nn.Module):
    """
    基础模块，ARTBlock，负责进行注意力计算等操作
    - 输入: (B, input_channels, H, W) 或 (B, input_channels, D, H, W)
    - 输出:
        * deep_supervision=False -> Tensor (B, num_classes, ...)
        * deep_supervision=True  -> list[Tensor], 第一个是最终输出，后面是辅助输出
    nnU-Net 训练器会根据 deep_supervision 来决定 loss 怎么算，所以这里要配合它。
    """
    def __init__(self,input_channels:int,num_class:int,deep_supervision:bool = True,**kwargs):
        super().__init__()
        self.input_channels = input_channels
        self.num_class = num_class
        self.decoder = _DecoderShim(deep_supervision)
        self.deep_supervision = False

        # 2d模块，后续可以通过if语句来添加上3d的模块
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,out_channels=32,kernel_size=3,padding=1),
            nn.InstanceNorm2d(32,affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,1,padding=1),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(32)
        )

        self.head = nn.Conv2d(32,num_class,1)
        self.aux_head = nn.Conv2d(32,num_class,1)

    def forward(self,x):
        if x.dim() != 4:
            raise RuntimeError(
                f"UNetArtBlock 的2d输入是（B，C，H，W)，但你给的是 shape={tuple(x.shape)}。"
            )
        feat = self.stem(x)
        out = self.head(feat)

        return out


