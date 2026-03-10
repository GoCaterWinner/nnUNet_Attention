import torch
import torch.nn as nn
from .art_block import  ART_block


class _DecoderShim:
    """
    nnU-Net 的训练脚本有时会试图访问 `network.decoder.deep_supervision`，
    为了防止其因为找不到属性而报错，我们在这里加上一个兼容垫片(Shim)。
    """
    def __init__(self, deep_supervision=True):
        self.deep_supervision = deep_supervision

class UNetARTBlock(nn.Module):
    """
    基础模块: ARTBlock，负责进行注意力计算等操作。
    本代码展示了如何将其兼容到 nnU-Net 的管道中。
    
    【输入维度控制】:
    nnU-Net 基于 plans 会传入 2D 或 3D 图像：
    - 2D: (B, input_channels, H, W) 
    - 3D: (B, input_channels, D, H, W)
    在设计自定义模块时，必须对这两种形状（或固定只容颜一种如果 plans 固定了结构）做兼容处理。
    
    【输出格式要求】:
    如果是配合 nnU-Net 默认训练跑：
    - deep_supervision=False -> 必须返回单一 Tensor (B, num_classes, ...)
    - deep_supervision=True  -> 必须返回 list[Tensor] ! 第一个是最高分辨率最终输出，后面依次是降采样后的辅助特征图输出。
    由于目前我们在 Trainer 里强行关闭了 DS，这里直接返回单个 Tensor 即可。
    """
    def __init__(self,input_channels:int,num_class:int,deep_supervision:bool = True,**kwargs):
        super().__init__()
        self.input_channels = input_channels
        self.num_class = num_class
        self.decoder = _DecoderShim(deep_supervision)
        self.deep_supervision = False

        # 示例 2D 骨干模块 (这里可以替换为你自己的 Attention 逻辑)
        # 后续可以通过 kwargs 中的 'dim' 或是 isinstance 判断，来动态选择 nn.Conv2d 或 nn.Conv3d
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
        """
        实际的前向传播位置：

        【输入 `x`】: 
            Tensor，形状通常为 (BatchSize, Input_Channels, H, W)
            在 nnU-Net 2D 任务中，BatchSize 取决于 plan 中规划的显存占用，H 和 W 为 Patch Size。
            
        【输出 `out`】:
            Tensor (如果 deep_supervision=False) 或者 List[Tensor] (如果 deep_supervision=True)。
            此例中 deep_supervision=False，直接返回单一 Tensor。
            形状为 (BatchSize, Num_Classes, H, W)。注意，这是未经过 Softmax 的 Logits 数据。
        """
        # 校验并强制验证 nnU-Net 的 batch 输入形状
        if x.dim() != 4:
            raise RuntimeError(
                f"UNetArtBlock 的2d输入是 (B, C, H, W)，但你给的是 shape={tuple(x.shape)}。如果遇到3D任务，请补充3D卷积模块。"
            )
        feat = self.stem(x)
        out = self.head(feat)

        # 这里如果 deep_supervision 返回 True，我们原本必须返回类似: [out, self.aux_head(feat_downsampled), ...]
        return out


