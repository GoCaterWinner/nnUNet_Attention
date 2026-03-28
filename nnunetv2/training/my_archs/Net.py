import torch
import torch.nn as nn
# 在这里给import过来
from nnunetv2.training.my_archs.MyBlock import MyBlock,WrappedStage


# 看这里！！这里就可以开始修改我们的某一层，也就是你熟知的“替换模块”或者说当”学术裁缝“了。
# 我们要做的事情，就是把原来的nnUNet为我们做好的网络拿过来，把其中的某一层或者多个层（论文中一般有ablation，所以我们可以采用堆工作量，2-3个modules）替换掉。
# 别担心，我在这里会给出具体的演示方法的，到时候你可以按照我的这个方法，进行替换即可。

# 假如我们关掉深监督
# 注意！！！这套逻辑针对我们的3d fullres，如果你换2d的，最好把逻辑换成conv2d，但是为了好的性能，3d的其实很不错了。
class YourNet(nn.Module):

    
    """
    模块流向图
----------------------------------------------------------------------------------------
Input              | (B, 1, 96, 160, 160)
                   | 输入 patch
Encoder Stage 0    | (B, 1, 96, 160, 160) --> (B, 32, 96, 160, 160)
                   | kernel=[3, 3, 3], stride=[1, 1, 1], blocks/convs=2
Encoder Stage 1    | (B, 32, 96, 160, 160) --> (B, 64, 48, 80, 80)
                   | kernel=[3, 3, 3], stride=[2, 2, 2], blocks/convs=2
Encoder Stage 2    | (B, 64, 48, 80, 80) --> (B, 128, 24, 40, 40)
                   | kernel=[3, 3, 3], stride=[2, 2, 2], blocks/convs=2
Encoder Stage 3    | (B, 128, 24, 40, 40) --> (B, 256, 12, 20, 20)
                   | kernel=[3, 3, 3], stride=[2, 2, 2], blocks/convs=2
Encoder Stage 4    | (B, 256, 12, 20, 20) --> (B, 320, 6, 10, 10)
                   | kernel=[3, 3, 3], stride=[2, 2, 2], blocks/convs=2
Bottleneck         | (B, 320, 6, 10, 10) --> (B, 320, 6, 5, 5)
                   | kernel=[3, 3, 3], stride=[1, 2, 2], blocks/convs=2
Decoder Stage 4    | spatial [6, 5, 5] --> [6, 10, 10]
                   | 输出通道=320, upsample=[1, 2, 2], convs=2
Decoder Stage 3    | spatial [6, 10, 10] --> [12, 20, 20]
                   | 输出通道=256, upsample=[2, 2, 2], convs=2
Decoder Stage 2    | spatial [12, 20, 20] --> [24, 40, 40]
                   | 输出通道=128, upsample=[2, 2, 2], convs=2
Decoder Stage 1    | spatial [24, 40, 40] --> [48, 80, 80]
                   | 输出通道=64, upsample=[2, 2, 2], convs=2
Decoder Stage 0    | spatial [48, 80, 80] --> [96, 160, 160]
                   | 输出通道=32, upsample=[2, 2, 2], convs=2
Seg Head           | (B, 32, 96, 160, 160) --> (B, 2, 96, 160, 160)
                   | 分割输出头
Deep Supervision   | enabled
                   | 多尺度监督开关
```
    """
    
    # 接下来请跳转到./nnunetv2/training/my_archs/MyBlock.py，我在那里写了如何建立模块进行替换的教程


    #————————————————————————————————————那里看完了不？继续————————————————————————————————————#

    def __init__(self,base_net):
        super().__init()
        self.base_net = base_net

        if hasattr(base_net,"decoder"):
            self.decoder = base_net.decoder

        # old_stage的作用来了哦！！
        # 比如我现在想替换Encoder Stage 2
        # 我之前做的可视化模型，里面记录着的列表，其实就是每一层的“名称”，拿到了名称，我们就可以制作替换层，从而达到替换模块的作用哦
        old_stage1 = self.base_net.encoder.stages[2]
        self.base_net.encoder.stages[2] = WrappedStage(old_stage=old_stage1, channels = 128)

        # 比如我还要再改一层，举个例子哦，这次我要改底层bottleneck，它的名字不叫做bottleneck哦，叫做stages[-1]
        # nnUNet是把底层算作encoder里面的，算最后一个
        old_stage2 = self.base_net.encoder.stages[-1]
        self.base_net.encoder.stages[-1] = WrappedStage(old_stage=old_stage2,channels = 320)
        # 为什么是128？打开我们的model_summary.md，你仔细看，经过这一层，之后，模型是不是变成了128？（也许你的不一样，但是我的这个是这个哈哈哈）
        # 也就是经过这个WrappedStage发生了这样一件事情，X ——> old——stage(也就是之前的第二层) ——> 你自己的模块MyBlock ——> out，这就完成了插拔模块。

    def forward(self,x):
        return self.base_net(x)

# 经过这个后，你的模块就成功了哦，再 pip install -e .再安装一次（貌似也不需要，其实只是我的习惯啦），然后就可以训练了哦，这就是完整的实验流程！

