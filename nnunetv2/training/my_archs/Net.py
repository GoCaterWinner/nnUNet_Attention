import torch
import torch.nn as nn

# 这个纯属没办法，nnUNetTrainer默认你的网络必须有个decoder属性，而decoder里面有个deep_supervision属性。
# 如果没有的话它就会报错，所以我们就加个假的decoder和deep_supervision属性，来让它不报错。
class _DecoderShim:
    """

    nnU-Net 的训练脚本有时会试图访问 `network.decoder.deep_supervision`，
    为了防止其因为找不到属性而报错，我们在这里加上一个兼容垫片(Shim)。
    这东西一定要留着，不然到时候网络接口没办法对齐
    """
    def __init__(self, deep_supervision=True):
        self.deep_supervision = deep_supervision

# 这里介绍一下几个输入，有点不一样的。
# nnUNet有四种模式，2d，3d_fullres，3d_lowres。2d的输入是(B, C, H, W)
# 3d_fullres的输入是(B, C, D, H, W)，3d_lowres的输入也是(B, C, D, H, W)，但是D通常比较小。
class YourNet(nn.Module):
    """

    这是一个示例网络，展示了如何构建一个符合 nnU-Net 的网络架构
    首先，如果你做3D的，输入是（B, C, D, H, W），如果是2D的，输入是（B, C, H, W）。

    """
    # 虽然默认值是 True，但是deep_supervision之前已经关闭掉了。
    # input_channels 是 nnU-Net 传入的参数，代表输入图像的通道数，比如CT通常是1，RGB图像是3。
    # num_class 是 nnU-Net 传入的参数，代表最终输出的类别数，比如二分类是2，多分类可能是4、5等。

    # **kwarg是字典型占位参数，就比如我举个例子哈。你调用YourNet（input_channels=1,num_class=2,deep_supervision=True,foo=123,bar='abc'）.
    # 那么kwargs就会打包出一个字典kwargs = {'foo': 123, 'bar': 'abc'}，这就很方便，比如你想给网络增加一些牛逼的特性。
    def __init__(self,input_channels:int,num_class:int,deep_supervision:bool = True,**kwargs):
        super().__init__()
        self.input_channels = input_channels
        self.num_class = num_class
        self.decoder = _DecoderShim(deep_supervision) # 垫片的作用，欺骗nnUNetTrainer
        self.deep_supervision = False

        # 我这里随便写一个比较简单的网络，大概你看一下意思，是这么搭建的，然后要改造的话，在我这个基础上改就行啦。
        # nn.Sequential是一个容器，可以把一系列层按顺序组合成一个模块。你在里面放的层会按照你放的顺序依次执行,是我最推荐的.也是最容易理解的一种搭建方法
        self.my_net = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=3, padding=1),  # 输入通道数，输出通道数，卷积核大小，填充
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.Conv3d(32, num_class, kernel_size=1)  # 最后输出 num_class 个通道，代表每个类别的预测
        )
       

    def forward(self,x):

        # ok,这就是一个最小的可以跑通的单元
        out = self.my_net(x)

        return out



