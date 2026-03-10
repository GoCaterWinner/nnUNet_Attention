import torch
import torch.nn as nn
from .art_block import  ART_block


class _DecoderShim:
    def __init__(self, deep_supervision=True):
        self.deep_supervision = deep_supervision

class UNetARTBlock(nn.Module):
    """
    基础模块，ARTBlock，负责进行注意力计算等操作。
    自动适配 2D 和 3D 输入：
    - 2D: (B, C, H, W)
    - 3D: (B, C, D, H, W)
    根据 nnU-Net plans 中传入的 conv_op 参数自动判断。
    """
    def __init__(self, input_channels: int, num_class: int, deep_supervision: bool = True, **kwargs):
        super().__init__()
        self.input_channels = input_channels
        self.num_class = num_class
        self.decoder = _DecoderShim(deep_supervision)
        self.deep_supervision = False

        # 根据 nnU-Net plans 中的 conv_op 自动判断是 2D 还是 3D 任务
        conv_op_str = kwargs.get('conv_op', 'torch.nn.modules.conv.Conv2d')
        if '3d' in conv_op_str.lower() or 'Conv3d' in conv_op_str:
            Conv = nn.Conv3d
            Norm = nn.InstanceNorm3d
            self.is_3d = True
        else:
            Conv = nn.Conv2d
            Norm = nn.InstanceNorm2d
            self.is_3d = False

        # 骨干模块（自动适配 2D/3D）
        self.stem = nn.Sequential(
            Conv(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1),
            Norm(32, affine=True),
            nn.ReLU(inplace=True),
            Conv(32, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            Norm(32)
        )

        self.head = Conv(32, num_class, 1)
        self.aux_head = Conv(32, num_class, 1)

    def forward(self, x):
        """
        前向传播：
        - 2D 输入: (B, C, H, W)
        - 3D 输入: (B, C, D, H, W)
        输出: (B, num_classes, ...) 未经 Softmax 的 Logits
        """
        expected_dim = 5 if self.is_3d else 4
        if x.dim() != expected_dim:
            raise RuntimeError(
                f"UNetArtBlock 期望 {'3D(B,C,D,H,W)' if self.is_3d else '2D(B,C,H,W)'} 输入，"
                f"但收到 shape={tuple(x.shape)}。"
            )
        feat = self.stem(x)
        out = self.head(feat)
        return out
