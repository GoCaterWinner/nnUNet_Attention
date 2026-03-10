from typing import Union, List, Tuple

import numpy as np
import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.my_archs.unet_art_block import UNetARTBlock
from nnunetv2.utilities.ccc_metric import compute_ccc


class MyTrainer_Attention(nnUNetTrainer):
    """
    nnU-Net 自定义模块添加教程：MyTrainer_Attention 示例

    【核心思想】：
    nnU-Net 的训练是由 `nnUNetTrainer` 控制的。要替换网络架构或修改损失函数，
    最标准的做法是继承 `nnUNetTrainer`，并重写（Override）相关的方法，如
    `build_network_architecture` 和 `_build_loss`。

    【运行逻辑链路】（当你输入 nnUNetv2_train ... -tr MyTrainer_Attention 时）：
    1. 命令行解析 `nnUNetv2_train` (在 nnunetv2/run/run_training.py) 提取参数。
    2. 发现 `-tr MyTrainer_Attention`，系统会在 `nnunetv2.training.nnUNetTrainer` 目录下搜索同名类。
    3. 加载对应的 `plans.json` 和 `dataset.json`。
    4. 实例化 `MyTrainer_Attention` 并调用其 `initialize()`。
    5. 在初始化过程中，相继调用 `_build_loss` 和 `build_network_architecture`。
    6. 加载 DataLoaders，最后执行 `run_training()` 开启循环。

    【重点关注的超参数/配置】：
    - self.enable_deep_supervision: 深度监督开关。nnU-Net 默认会对网络多尺度输出算 Loss。如果你的模块没有多尺度输出，必须置 False。
    - self.configuration_manager.batch_dice: 是否在整个 batch 级别计算 Dice（针对小目标稳定训练）。
    - 学习率、优化器 (如果需要修改可重写 `configure_optimizers`)。

    【CCC 体积指标】：
    - 在每个 epoch 的验证阶段，除了计算 pseudo Dice，还会计算 CCC（一致性相关系数）。
    - CCC 衡量预测体积（体素数）与真实体积的一致性，值域 [-1, 1]，越接近 1 越好。
    - CCC 会被记录到 logger 并打印到训练日志，便于监控脂肪分割的体积准确性。
    """

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True, device=None):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # 注意超参：深度监督(Deep Supervision)总开关先关掉，因为我们的自定义模块可能没做多尺度输出
        self.enable_deep_supervision = False

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        覆盖父类方法：无论父类怎么设置，强制不让父类去碰 self.network.decoder.deep_supervision，
        因为我们的自定义网络（如 UNetARTBlock）目前只输出单独一个张量。
        如果你的注意力模块支持输出多尺度特征图（List[Tensor]），则可以开启此选项。
        """
        self.enable_deep_supervision = False

    def _build_loss(self):
        """
        覆盖父类的 Loss 构建：
        因为关闭了深度监督（Deep Supervision），我们需要返回普通的 Loss。
        原版 nnU-Net 默认在有深度监督时会套一层 `DeepSupervisionWrapper`。

        【输入】: 无直接参数传递（但该返回的 loss 函数对象在训练时接收：
                 网络预测 logits 如 (B, Class, H, W), 和真实标签 target (B, 1, H, W)）
        【输出】: nn.Module（例如 DC_and_CE_loss 对象，调用时返回 Scalar 标量损失值）
        """
        if self.label_manager.has_regions:
            # region-based（比如 BraTS 任务中的重叠区域，少见）
            from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
            loss = DC_and_BCE_loss(
                {},
                {'batch_dice': self.configuration_manager.batch_dice,
                 'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=None
            )
        else:
            # 绝大多数多分类任务走这里
            from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
            loss = DC_and_CE_loss(
                {'batch_dice': self.configuration_manager.batch_dice,
                 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                {},
                weight_ce=1, weight_dice=1,  # 这里可以修改 CE 和 Dice 的权重比例
                ignore_label=self.label_manager.ignore_label,
                dice_class=None
            )
        return loss

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,  # 重点超参：网络输入通道数 (基于 modality 数量)
                                   num_output_channels: int,  # 重点超参：网络输出通道数 (几分类)
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        【模块修改的核心位置】
        在这里注入你自定义的网络架构（例如含 Attention 的网络）。
        我们将原先可能基于 plans 生成的默认架构替换为你手写的 `UNetARTBlock`。

        【输入】:
            - num_input_channels (int): 输入图像通道数（例如模态数：1 代表单模态 MR/CT, 3 代表 RGB）
            - num_output_channels (int): 输出通道数（例如分类数量：包含背景的分割目标数）
            - enable_deep_supervision (bool): 是否启用了深度监督
        【输出】: nn.Module (你自定义的、可以进行 forward 运算的 PyTorch 网络实例)
        """
        print("MyTrainer_Attention build_network_architecture called")  # 确保函数正确调用
        print("\n num_input_channels:", num_input_channels)
        print("\n num_output_channels:", num_output_channels)
        print("\n enable_deep_supervision:", enable_deep_supervision)
        print("\n (ignored) architecture_class_name:", architecture_class_name)
        print("\n (ignored) arch_init_kwargs keys:", list(arch_init_kwargs.keys()))

        # 实例化我们自己的网络
        # 先显性修改函数签名，强制关闭内置的 deep_supervision
        net = UNetARTBlock(
            input_channels=num_input_channels,
            num_class=num_output_channels,
            deep_supervision=False,
            **arch_init_kwargs
        )

        return net

    # =========================================================================
    # 以下是新增的 CCC 体积指标重写方法
    # =========================================================================

    def validation_step(self, batch: dict) -> dict:
        """
        在父类 validation_step 基础上，额外返回每个类别的前景体素数量，
        用于后续在 on_validation_epoch_end 中计算 CCC。

        注意：这里计算的是 patch 级别的体素数，是 "pseudo CCC"，
        和完整推断后的 CCC 含义一致但数值上略有差异（类似 pseudo Dice）。
        """
        from nnunetv2.utilities.helpers import dummy_context
        from torch.amp import autocast
        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        # 深度监督时只取最高分辨率
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float16)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()

        if not self.label_manager.has_regions:
            # [1:] 去掉背景类
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        # ---- 新增：计算体素体积用于 CCC ----
        # vol_pred = 预测的前景体素数（tp + fp），vol_ref = 真实的前景体素数（tp + fn）
        vol_pred = (tp_hard + fp_hard)   # shape: (num_fg_classes,)
        vol_ref  = (tp_hard + fn_hard)   # shape: (num_fg_classes,)

        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
            'vol_pred': vol_pred,   # 每个前景类别的预测体积（体素数）
            'vol_ref':  vol_ref,    # 每个前景类别的真实体积（体素数）
        }

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        """
        在父类逻辑（计算 mean Dice）基础上，额外计算 pseudo CCC 并记录到 logger。
        """
        from nnunetv2.utilities.collate_outputs import collate_outputs
        import torch.distributed as dist

        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        # 收集体积数组：shape (num_batches, num_fg_classes)
        all_vol_pred = np.array(outputs_collated['vol_pred'])  # (N_steps, C)
        all_vol_ref  = np.array(outputs_collated['vol_ref'])   # (N_steps, C)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()

            # 汇聚各 GPU 上的体积数据
            vp_list = [None for _ in range(world_size)]
            dist.all_gather_object(vp_list, all_vol_pred)
            all_vol_pred = np.vstack(vp_list)

            vr_list = [None for _ in range(world_size)]
            dist.all_gather_object(vr_list, all_vol_ref)
            all_vol_ref = np.vstack(vr_list)
        else:
            loss_here = np.mean(outputs_collated['loss'])

        # ---- Dice（与父类相同）----
        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

        # ---- CCC（新增）----
        # all_vol_pred/ref shape: (N_steps, num_fg_classes)
        # 对每个前景类别独立计算 CCC，然后取均值作为 pseudo mean CCC
        num_fg_classes = all_vol_pred.shape[1] if all_vol_pred.ndim == 2 else 1
        ccc_per_class = []
        for c in range(num_fg_classes):
            vp_c = all_vol_pred[:, c] if all_vol_pred.ndim == 2 else all_vol_pred
            vr_c = all_vol_ref[:, c]  if all_vol_ref.ndim == 2  else all_vol_ref
            ccc_per_class.append(compute_ccc(vr_c, vp_c))

        mean_ccc = float(np.nanmean(ccc_per_class))
        self.logger.log('val_ccc', mean_ccc, self.current_epoch)

    def on_epoch_end(self):
        """
        在父类的打印逻辑基础上，额外打印 CCC 值。
        """
        self.logger.log('epoch_end_timestamps', __import__('time').time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss',   np.round(self.logger.my_fantastic_logging['val_losses'][-1],   decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])

        # ---- 新增：打印 CCC ----
        val_ccc = self.logger.my_fantastic_logging['val_ccc'][-1]
        self.print_to_log_file(f'val_CCC (volume accuracy): {np.round(val_ccc, decimals=4)}   '
                               f'[CCC 接近 1.0 = 体积预测越准确，临床建议 > 0.9]')

        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # 周期性保存 checkpoint
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(__import__('os.path', fromlist=['join']).join(self.output_folder, 'checkpoint_latest.pth'))

        # 保存最佳 checkpoint（基于 ema Dice）
        from batchgenerators.utilities.file_and_folder_operations import join
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1
