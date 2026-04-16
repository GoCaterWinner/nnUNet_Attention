import torch
import torch.nn as nn
from .Blockrepo import Mlp, window_partition, window_reverse, SwinTransformerBlock,PatchEmbed


# 我在这里举个例子，比如我这里做一个可插拔模块，我这里选择做一个最简单的，来起到演示作用。
# 第一步，我们先写一个小模块，比如这个叫做MyBlock。
class MyBlock(nn.Module):
    def __init__ (self,channels):
        super().__init__()

        # 以后你包装模块的时候，我很推荐你用Sequential容器把这个模块的积木保存好，这样子逻辑就很顺
        self.block = nn.Sequential(
            nn.Conv3d(channels,channels,kernel_size = 1),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.block(x)
    
# 做一个包装层。
class WrappedStage(nn.Module):
    def __init__(self,old_stage,channels):
        super().__init__()
        # old_stage的意思就是原来的模块
        self.old_stage = old_stage
        # 在这里给它实例化了
        self.my_block = MyBlock(channels)

    def forward(self,x):
        # 这个代码的意思就是告诉你，我在原来的模块后面，接上了我自己的模块
        x = self.old_stage(x)
        x = self.my_block(x)
        return x
    
# 继续回到Net.py,我们往下看

class TransformerBottleneck(nn.Module):
    def __init__(self, old_stage, channels):
        super().__init__()
        self.old_stage = old_stage
        self.channels = channels
        # 这里按你当前 3d_fullres 的 bottleneck 实际尺寸写死。
        # 你的模型检查报告里 bottleneck 是 (B, 320, 6, 5, 5)，
        # 所以这里直接固定成 (6, 5, 5)，这样 window_size=2 不会被偷偷改成 1。
        self.input_resolution = (6, 5, 5)
        self.window_size = 2
        self.transformer_block = SwinTransformerBlock(
            dim=channels,
            input_resolution=self.input_resolution,
            num_heads=8,
            window_size=self.window_size
        )

    def forward(self, x):
        # old_stage 输出仍然是卷积特征图
        x = self.old_stage(x)   # shape: (B, C, D, H, W)

        B, C, D, H, W = x.shape

        if C != self.channels:
            raise RuntimeError(
                f"TransformerBottleneck 预期输入通道为 {self.channels}，但实际拿到的是 {C}。"
            )
        if (D, H, W) != self.input_resolution:
            raise RuntimeError(
                "TransformerBottleneck 当前是按 3d_fullres bottleneck 专用版本写的，"
                f"预期空间尺寸是 {self.input_resolution}，但实际拿到的是 {(D, H, W)}。"
            )

        # Conv feature -> Transformer token
        x = x.permute(0, 2, 3, 4, 1).contiguous()   # (B, D, H, W, C)
        x = x.view(B, D * H * W, C)                 # (B, N, C), N = D*H*W

        # 进入 transformer
        x = self.transformer_block(x, None)         # (B, N, C)

        # Transformer token -> Conv feature
        x = x.view(B, D, H, W, C)                   # (B, D, H, W, C)
        x = x.permute(0, 4, 1, 2, 3).contiguous()   # (B, C, D, H, W)

        return x
