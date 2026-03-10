from typing import Union, List, Tuple

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.my_archs.unet_art_block import UNetARTBlock


class MyTrainer_Attention(nnUNetTrainer):
    """
    自定义 Trainer：在 nnUNetTrainer 基础上替换网络为 UNetARTBlock（含 Attention），
    并关闭 Deep Supervision。

    CCC 指标已在父类 nnUNetTrainer 中集成，无需在此重复。
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False

    def set_deep_supervision_enabled(self, enabled: bool):
        self.enable_deep_supervision = False

    def _build_loss(self):
        """
        不使用 DeepSupervisionWrapper，直接返回普通 loss。
        """
        from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

        if self.label_manager.has_regions:
            from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
            loss = DC_and_BCE_loss(
                {},
                {'batch_dice': self.configuration_manager.batch_dice,
                 'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss
            )
        else:
            from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
            loss = DC_and_CE_loss(
                {'batch_dice': self.configuration_manager.batch_dice,
                 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                {},
                weight_ce=1, weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss
            )
        return loss

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        print("MyTrainer_Attention build_network_architecture called")
        print("\n num_input_channels:", num_input_channels)
        print("\n num_output_channels:", num_output_channels)
        print("\n enable_deep_supervision:", enable_deep_supervision)
        print("\n (ignored) architecture_class_name:", architecture_class_name)
        print("\n (ignored) arch_init_kwargs keys:", list(arch_init_kwargs.keys()))

        net = UNetARTBlock(
            input_channels=num_input_channels,
            num_class=num_output_channels,
            deep_supervision=False,
            **arch_init_kwargs
        )
        return net
