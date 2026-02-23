from typing import List, Tuple, Union

import torch
from torch import nn
from torch._dynamo import OptimizedModule

from nnunetv2.training.my_archs.unet_art_block import UNetARTBlock
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


class MyTrainer_Attention(nnUNetTrainer):
    """
    在 nnUNetTrainer 基础上，插入 ART 注意力模块的训练器。

    关键思路（尽量保持与原框架接口对齐）：
    1. 网络主干仍使用 plans 里定义的 nnU-Net 原生结构；
    2. 在主干输出端增加 ART 细化层（UNetARTBlock 包装器）；
    3. deep supervision 的开关仍走 nnU-Net 原有流程，保证 train/val/infer 一致。
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device("cuda")):
        """
        这里的参数签名必须与父类 nnUNetTrainer.__init__ 对齐（至少不能多出父类没有的参数）。

        原因：
        - 父类 __init__ 内部会用 inspect.signature(self.__init__) 读取“当前子类”的签名；
        - 然后按参数名从父类 __init__ 的 locals() 里逐个取值保存；
        - 如果子类多了一个参数（例如 unpack_dataset），父类 locals() 里没有这个名字，就会在
          nnUNetTrainer.py:112 的 locals()[k] 处触发 KeyError。

        所以这里严格保持与 run_training.py 的实例化调用一致，避免接口不对齐导致初始化失败。
        """
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        # 保持与 nnU-Net 默认行为一致：默认开启 deep supervision
        self.enable_deep_supervision = True

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        这是 nnU-Net 在训练和推理时都会调用的建网入口。
        这里必须严格对齐签名，否则 `nnUNetv2_train ... -tr MyTrainer_Attention` 会找得到类但跑不起来。
        """
        # 1) 先按 plans 构建原生 nnU-Net 主干，确保 2D/3D、通道数、stage 配置全部沿用官方逻辑
        backbone = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision
        )
        # 2) 再把 ART 模块以“包装器”的方式缝在输出端，做到低侵入接入
        network = UNetARTBlock(
            backbone=backbone,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision
        )
        return network

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        覆盖父类的 deep supervision 开关逻辑，适配“包装网络”的层级结构。

        为什么要重写：
        - 父类默认写法会直接访问 `network.decoder.deep_supervision`；
        - 我们现在的网络是 `UNetARTBlock(backbone)`，需要同步两层：
          1) 包装层自己的 decoder.deep_supervision
          2) 内部 backbone.decoder.deep_supervision
        """
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        if hasattr(mod, "set_deep_supervision"):
            mod.set_deep_supervision(enabled)
        elif hasattr(mod, "decoder") and hasattr(mod.decoder, "deep_supervision"):
            # 兜底逻辑：如果未来替换成其他网络，至少不至于直接崩
            mod.decoder.deep_supervision = enabled
