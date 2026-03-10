from typing import Union, List, Tuple

import numpy as np
import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.my_archs.unet_art_block import UNetARTBlock
from nnunetv2.utilities.ccc_metric import compute_ccc


class MyTrainer_Attention(nnUNetTrainer):
    """
    自定义 Trainer：在 nnUNetTrainer 基础上
    1. 替换网络为 UNetARTBlock（含 Attention）
    2. 关闭 Deep Supervision
    3. 训练过程中计算 pseudo CCC（体积一致性），训练结束后由 evaluate_predictions 计算完整 CCC
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False

    def set_deep_supervision_enabled(self, enabled: bool):
        # 强制关闭 deep supervision
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

    # =========================================================================
    # 以下是 CCC 体积指标相关的重写方法
    # =========================================================================

    def validation_step(self, batch: dict) -> dict:
        """
        在父类 validation_step 基础上，额外返回每个类别的前景体素数量，
        用于后续在 on_validation_epoch_end 中计算 pseudo CCC。
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
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        # ---- 新增：计算体素体积用于 CCC ----
        vol_pred = (tp_hard + fp_hard)   # shape: (num_fg_classes,)
        vol_ref  = (tp_hard + fn_hard)   # shape: (num_fg_classes,)

        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
            'vol_pred': vol_pred,
            'vol_ref':  vol_ref,
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

        # 收集体积数组
        all_vol_pred = np.array(outputs_collated['vol_pred'])
        all_vol_ref  = np.array(outputs_collated['vol_ref'])

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
        self.print_to_log_file(f'Pseudo CCC (volume accuracy): {np.round(val_ccc, decimals=4)}   '
                               f'[CCC 接近 1.0 = 体积预测越准确，临床建议 > 0.9]')

        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # 周期性保存 checkpoint
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # 保存最佳 checkpoint（基于 ema Dice）
        from batchgenerators.utilities.file_and_folder_operations import join
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1
