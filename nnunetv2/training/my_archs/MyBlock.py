import torch
import torch.nn as nn


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