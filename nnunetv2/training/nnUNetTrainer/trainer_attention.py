from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn

from nnunetv2.training.my_archs.unet_art_block import UNetARTBlock
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.ccc_metric import compute_ccc
from nnunetv2.utilities.hd95_metric import compute_hd95


class AttentionOrLinearPlaceHolder(nn.Module):
    """
    这是一个“教学占位模块”，默认不会被接入当前训练图。

    你以后如果想往网络里添加这些结构，就可以参考这个壳子自己实现：

    - `nn.Linear`
    - `nn.MultiheadAttention`
    - `MLP`
    - `SE / CBAM / 自定义注意力`
    - `特征融合分支`

    重要：
    这个类的 `pass` 是故意保留的，因为它只是一个模板，不参与当前训练。
    真正要把模块接进网络时，请去改 `nnunetv2/training/my_archs/unet_art_block.py`，
    然后再在 `MyTrainer_Attention.build_network_architecture(...)` 里返回你的新网络。
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        """
        参数说明
        ----------
        in_features:
            输入特征维度。
            如果你前面已经把卷积特征展平成 token 或向量，这里通常对应最后一维大小。

        hidden_features:
            中间隐层维度。
            如果你想做两层 MLP、门控单元或者 attention 前后的投影，这通常是中间通道数。

        out_features:
            输出特征维度。
            一般需要和后续模块能对上，比如回到原通道数，或者投影到新的 embedding 维度。
        """
        super().__init__()

        # 以后你可以按下面这种形式真正实现:
        #
        # self.fc1 = nn.Linear(in_features, hidden_features)
        # self.act = nn.GELU()
        # self.fc2 = nn.Linear(hidden_features, out_features)
        #
        # 或者:
        # self.attn = nn.MultiheadAttention(embed_dim=in_features, num_heads=8, batch_first=True)
        #
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数说明
        ----------
        x:
            输入特征张量。形状完全取决于你自己的设计。
            常见情况包括：
            - `(B, N, C)`：token 序列
            - `(B, C)`：全局池化后的向量
            - `(B, C, H, W)` / `(B, C, D, H, W)`：卷积特征图

        返回
        ----------
        torch.Tensor:
            你的模块处理后的结果。

        这里保留 `pass`，表示“等你真正实现时再填”。
        """
        pass


class MyTrainer_Attention(nnUNetTrainer):
    """
    这是一个给“模型层二次开发”准备的 trainer 模板。

    你可以把它理解成 nnU-Net 和你自定义网络之间的胶水层，它主要负责四件事：

    1. 接住 nnU-Net 传进来的 plans / dataset / fold / device 等上下文信息。
    2. 告诉 nnU-Net 到底该实例化哪一个网络。
    3. 告诉 nnU-Net 用什么 loss、deep supervision、optimizer。
    4. 如果需要，追加你自己的验证指标和日志。

    你当前这条开发路线非常推荐：

    - 数据处理层不动
    - dataloader 不动
    - 训练框架主循环不动
    - 只在 trainer 层和 network 层做模型改造

    当你执行：

    `nnUNetv2_train DATASET_ID 2d 0 -tr MyTrainer_Attention`

    大致函数链是：

    `run_training_entry()`
    -> `run_training(...)`
    -> `get_trainer_from_args(...)`
    -> 查找到 `MyTrainer_Attention`
    -> 实例化 trainer
    -> `run_training()`
    -> `initialize()`
    -> `build_network_architecture(...)`
    -> `_build_loss()`
    -> `get_dataloaders()`
    -> `train_step()` / `validation_step()`

    真正模型被调用的位置不是 `build_network_architecture(...)`，
    而是 `train_step()` / `validation_step()` 里面的：

    `output = self.network(data)`
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        """
        参数说明
        ----------
        plans:
            nnU-Net 规划器生成的总配置字典。
            它来自 `nnUNet_preprocessed/<DatasetName>/<plans_identifier>.json`。

            这里面包含的不是“单纯网络参数”，而是整套实验规划信息，例如：
            - 不同 configuration 的定义
            - patch size
            - batch size
            - network class name
            - network init kwargs
            - spacing / resampling / cascade 相关信息

            父类会把它包装成 `PlansManager`，后续很多接口都从这里读配置。

        configuration:
            当前训练配置名，通常是：
            - `2d`
            - `3d_fullres`
            - `3d_lowres`
            - `3d_cascade_fullres`

            它会决定当前训练到底选用哪套 patch size、batch size、network kwargs。

        fold:
            当前交叉验证 fold。
            常见取值是 `0~4`，也可能是 `all`。

            这个参数最终会影响：
            - 数据划分
            - 输出目录
            - checkpoint 位置

        dataset_json:
            预处理目录中的 `dataset.json` 内容。
            它会告诉 nnU-Net：
            - label 定义
            - file ending
            - channel / modality 信息
            - 是否有 ignore label

            父类会基于它构造 `label_manager`。

        device:
            当前训练设备。
            常见是：
            - `torch.device("cuda")`
            - `torch.device("cpu")`
            - `torch.device("mps")`

            父类还会根据 DDP 情况重新修正实际使用的 device。
        """
        super().__init__(plans, configuration, fold, dataset_json, device)

        # 如果你的网络不是“多尺度输出列表”，而是只输出一个 segmentation logits 张量，
        # 最稳妥的做法就是直接关闭 deep supervision。
        self.enable_deep_supervision = False

    def configure_optimizers(self):
        """
        这个接口控制“优化器”和“学习率调度器”。

        你以后如果想改这些内容，通常就在这里改：
        - SGD -> Adam / AdamW
        - PolyLR -> CosineAnnealing / Warmup / OneCycle
        - 学习率、weight decay、momentum

        当前先保持父类默认行为，避免影响现有训练流程。

        返回
        ----------
        Tuple[torch.optim.Optimizer, object]:
            第一个返回值是 optimizer，第二个返回值是 lr scheduler。
        """
        return super().configure_optimizers()

    def train_step(self, batch: dict) -> dict:
        """
        这是“一次训练迭代”的执行入口。

        参数说明
        ----------
        batch:
            dataloader 产出的一个 batch，通常包含：
            - `batch["data"]`
            - `batch["target"]`

            其中常见形状为：
            - 2D: `data.shape == (B, C, H, W)`
            - 3D: `data.shape == (B, C, D, H, W)`

        返回
        ----------
        dict:
            父类默认会返回一个字典，至少包含：
            - `loss`

        最重要的一点
        ----------
        真正调用模型 forward 的地方就在父类这个接口里：

        `output = self.network(data)`

        所以如果你只是在改网络结构，通常不用改这里，
        只需要保证 `build_network_architecture(...)` 返回的网络能吃下这个 `data` 即可。
        """
        return super().train_step(batch)

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        这个接口控制“是否在网络内部启用 deep supervision”。

        参数说明
        ----------
        enabled:
            父类希望写进去的 deep supervision 开关。
            在原版 nnU-Net 默认网络中，这通常会去修改：
            `network.decoder.deep_supervision`

        为什么这里要重写
        ----------
        因为你的自定义网络不一定有：
        - `decoder`
        - `decoder.deep_supervision`
        - 多尺度输出列表

        所以最安全的做法是重写这个函数，直接忽略父类的切换请求，
        始终强制保持：

        `self.enable_deep_supervision = False`

        这样可以避免：
        - 父类访问不存在的 `decoder`
        - loss 端误以为输出是 `List[Tensor]`
        - 训练和验证阶段的输出形状不一致
        """
        self.enable_deep_supervision = False

    def _build_loss(self):
        """
        这个接口负责构造训练使用的 loss。

        它没有显式函数参数，但会依赖 trainer 当前状态中的这些成员：
        - `self.label_manager`
        - `self.configuration_manager.batch_dice`
        - `self.label_manager.ignore_label`
        - `self.is_ddp`
        - `self.enable_deep_supervision`

        当前策略
        ----------
        因为我们已经关闭了 deep supervision，所以这里返回的是“普通 loss”，
        而不是父类那种再额外套一层 `DeepSupervisionWrapper` 的版本。

        返回
        ----------
        nn.Module:
            一个可调用的 loss 对象。
            训练时会被这样使用：

            `loss = self.loss(output, target)`

        你以后最常改的地方
        ----------
        - Dice + CE 的权重
        - 是否改成 BCE
        - 是否加 focal / boundary / topology loss
        - 是否改 region-based training 的分支逻辑
        """
        from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

        if self.label_manager.has_regions:
            from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss

            loss = DC_and_BCE_loss(
                {},
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "do_bg": True,
                    "smooth": 1e-5,
                    "ddp": self.is_ddp,
                },
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
        else:
            from nnunetv2.training.loss.compound_losses import DC_and_CE_loss

            loss = DC_and_CE_loss(
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "smooth": 1e-5,
                    "do_bg": False,
                    "ddp": self.is_ddp,
                },
                {},
                weight_ce=1,
                weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )

        return loss

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        """
        这是“模型结构接入 nnU-Net”的核心接口。

        参数说明
        ----------
        architecture_class_name:
            plans 里记录的原始网络类名。
            在原版 nnU-Net 里，这个参数会决定默认架构如何被构建。

            但在你当前这个自定义 trainer 中，我们并不直接依赖它来实例化网络，
            因为我们要手动 `return UNetARTBlock(...)`。

            它保留在函数签名里，主要是为了兼容 nnU-Net 的统一调用方式。

        arch_init_kwargs:
            plans 中为网络准备的初始化参数字典。
            常见内容包括：
            - `n_stages`
            - `features_per_stage`
            - `conv_op`
            - `kernel_sizes`
            - `strides`
            - `n_conv_per_stage`
            - `n_conv_per_stage_decoder`

            这些参数非常有价值，因为它们已经根据 nnU-Net 的实验规划自动适配过。
            如果你的自定义网络仍然是 U-Net 风格，多数情况下建议继续利用这些参数。

        arch_init_kwargs_req_import:
            某些 kwargs 对应的类或函数需要动态导入时，nnU-Net 会用到这个字段。

            对你当前这个手写网络来说，它通常不是最核心的参数，
            但函数签名必须保留，才能和父类的调用规范对齐。

        num_input_channels:
            网络输入通道数。
            它不是你手工写死的，而是 nnU-Net 根据数据集自动推出来的。

            例如：
            - 单模态 CT/MR 可能是 `1`
            - 多模态输入可能是 `2`、`4` 等

            你自定义网络第一层一定要和它对齐。

        num_output_channels:
            网络输出通道数，也就是 segmentation heads 数量。
            这个值由 `label_manager` 决定，不一定等于“类别总数”本身。

            原因是 nnU-Net 支持：
            - 普通 class-based segmentation
            - region-based segmentation
            - ignore label

            所以最稳的做法就是永远直接使用这个参数，不要自己手写类别数。

        enable_deep_supervision:
            父类传进来的 deep supervision 开关。
            注意它只是“父类当前的配置意图”，不代表你一定要照做。

            如果你的网络只输出单尺度 logits，那就应该像当前这样直接忽略它，
            并在真正实例化网络时写死 `deep_supervision=False`。

        返回
        ----------
        nn.Module:
            一个真正可执行 forward 的 PyTorch 网络对象。

        你以后最常改的地方
        ----------
        这里就是你替换网络的主入口。

        常见改法：
        - `return UNetARTBlock(...)`
        - `return YourUNetWithAttention(...)`
        - `return YourHybridCNNTransformer(...)`

        关键理解
        ----------
        这里做的是“构造模型对象”，不是“执行模型推理”。
        真正 forward 的调用发生在：
        `train_step()` / `validation_step()` 里的 `self.network(data)`。
        """
        print("MyTrainer_Attention build_network_architecture called")
        print("\nnum_input_channels:", num_input_channels)
        print("\nnum_output_channels:", num_output_channels)
        print("\nenable_deep_supervision from parent:", enable_deep_supervision)
        print("\nignored architecture_class_name:", architecture_class_name)
        print("\narch_init_kwargs keys:", list(arch_init_kwargs.keys()))
        print("\narch_init_kwargs_req_import:", arch_init_kwargs_req_import)

        # 这里是当前工程真正接入网络的位置。
        # 如果你未来写了自己的网络，比如在 `unet_art_block.py` 里加入了：
        # - nn.Linear
        # - 自注意力
        # - MLP
        # - 跨尺度融合
        # 那么最终通常就是在这里把返回对象替换掉。
        network = UNetARTBlock(
            input_channels=num_input_channels,
            num_class=num_output_channels,
            deep_supervision=False,
            **arch_init_kwargs,
        )
        return network

    def validation_step(self, batch: dict) -> dict:
        """
        这是“一次验证迭代”的执行入口。

        参数说明
        ----------
        batch:
            dataloader 产出的一个验证 batch。
            与 `train_step(...)` 一样，内部通常包含：
            - `batch["data"]`
            - `batch["target"]`

        返回
        ----------
        dict:
            除了父类常规的：
            - `loss`
            - `tp_hard`
            - `fp_hard`
            - `fn_hard`

            这里还额外返回：
            - `vol_pred`
            - `vol_ref`

            这样做是为了在 `on_validation_epoch_end(...)` 中进一步计算 CCC。

        说明
        ----------
        这里的 CCC 不是基于完整 case 推理结果算的，而是 patch-level 的 online 统计，
        所以更准确地说它是 pseudo CCC。

        这里新增的 HD95 也是同样口径的 pseudo HD95：
        它基于 validation patch 和预处理后的 configuration spacing，
        仅用于训练过程中的在线监控，不等于最终整例推理得到的真实 case-level HD95。
        """
        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
        from nnunetv2.utilities.helpers import dummy_context
        from torch.amp import autocast

        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            del data
            loss = self.loss(output, target)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output.shape, device=output.device, dtype=torch.float16
            )
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

        hd95_per_case = self._compute_batch_hd95(
            predicted_segmentation_onehot=predicted_segmentation_onehot,
            target=target,
            valid_mask=mask,
        )

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target, axes=axes, mask=mask
        )

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()

        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        vol_pred = tp_hard + fp_hard
        vol_ref = tp_hard + fn_hard

        return {
            "loss": loss.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
            "vol_pred": vol_pred,
            "vol_ref": vol_ref,
            "hd95": hd95_per_case,
        }

    def _compute_batch_hd95(
        self,
        predicted_segmentation_onehot: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Union[torch.Tensor, None],
    ) -> np.ndarray:
        """
        Compute patch-level pseudo HD95 for each sample and foreground class.

        返回数组形状：
        - conventional training: `(B, num_foreground_classes)`
        - region-based training: `(B, num_regions)`
        """
        if target.ndim != predicted_segmentation_onehot.ndim:
            target = target.view((target.shape[0], 1, *target.shape[1:]))

        batch_size = predicted_segmentation_onehot.shape[0]
        if self.label_manager.has_regions:
            class_indices = list(range(predicted_segmentation_onehot.shape[1]))
        else:
            class_indices = list(range(1, predicted_segmentation_onehot.shape[1]))

        hd95_per_case = np.full((batch_size, len(class_indices)), np.nan, dtype=np.float64)
        spacing = tuple(float(i) for i in self.configuration_manager.spacing)

        predicted_np = predicted_segmentation_onehot.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        valid_mask_np = valid_mask.detach().cpu().numpy().astype(bool) if valid_mask is not None else None

        for batch_index in range(batch_size):
            valid_here = valid_mask_np[batch_index, 0] if valid_mask_np is not None else None
            for output_index, class_index in enumerate(class_indices):
                pred_mask = predicted_np[batch_index, class_index].astype(bool)
                if self.label_manager.has_regions:
                    ref_mask = target_np[batch_index, class_index].astype(bool)
                else:
                    ref_mask = target_np[batch_index, 0] == class_index

                if valid_here is not None:
                    pred_mask = pred_mask & valid_here
                    ref_mask = ref_mask & valid_here

                hd95_per_case[batch_index, output_index] = compute_hd95(ref_mask, pred_mask, spacing)

        return hd95_per_case

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        """
        这个接口负责把整个验证 epoch 的结果聚合起来。

        参数说明
        ----------
        val_outputs:
            一个列表，里面的每个元素都是 `validation_step(...)` 返回的字典。

        这里做的事
        ----------
        - 聚合所有 step 的 `tp/fp/fn`
        - 计算 pseudo Dice
        - 聚合 `vol_pred/vol_ref`
        - 计算每个类别的 pseudo CCC
        - 计算每个类别的 pseudo HD95
        - 把结果写入 logger

        什么时候你会改这里
        ----------
        - 你想加 epoch 级别的新指标
        - 你想把多个 step 的统计量做统一汇总
        - 你想额外记录 per-class 指标
        """
        import torch.distributed as dist
        from nnunetv2.utilities.collate_outputs import collate_outputs

        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated["tp_hard"], 0)
        fp = np.sum(outputs_collated["fp_hard"], 0)
        fn = np.sum(outputs_collated["fn_hard"], 0)

        all_vol_pred = np.array(outputs_collated["vol_pred"])
        all_vol_ref = np.array(outputs_collated["vol_ref"])
        all_hd95 = np.array(outputs_collated["hd95"])

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
            dist.all_gather_object(losses_val, outputs_collated["loss"])
            loss_here = np.vstack(losses_val).mean()

            vp_list = [None for _ in range(world_size)]
            dist.all_gather_object(vp_list, all_vol_pred)
            all_vol_pred = np.vstack(vp_list)

            vr_list = [None for _ in range(world_size)]
            dist.all_gather_object(vr_list, all_vol_ref)
            all_vol_ref = np.vstack(vr_list)

            hd95_list = [None for _ in range(world_size)]
            dist.all_gather_object(hd95_list, all_hd95)
            all_hd95 = np.vstack(hd95_list)
        else:
            loss_here = np.mean(outputs_collated["loss"])

        global_dc_per_class = [
            i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]
        ]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log("mean_fg_dice", mean_fg_dice, self.current_epoch)
        self.logger.log("dice_per_class_or_region", global_dc_per_class, self.current_epoch)
        self.logger.log("val_losses", loss_here, self.current_epoch)

        num_fg_classes = all_vol_pred.shape[1] if all_vol_pred.ndim == 2 else 1
        ccc_per_class = []
        for class_index in range(num_fg_classes):
            vol_pred_class = all_vol_pred[:, class_index] if all_vol_pred.ndim == 2 else all_vol_pred
            vol_ref_class = all_vol_ref[:, class_index] if all_vol_ref.ndim == 2 else all_vol_ref
            ccc_per_class.append(compute_ccc(vol_ref_class, vol_pred_class))

        mean_ccc = float(np.nanmean(ccc_per_class))
        self.logger.log("val_ccc", mean_ccc, self.current_epoch)

        if all_hd95.ndim == 3:
            all_hd95 = all_hd95.reshape(-1, all_hd95.shape[-1])
        mean_hd95 = float(np.nanmean(all_hd95))
        self.logger.log("val_hd95", mean_hd95, self.current_epoch)

    def on_epoch_end(self):
        """
        这个接口负责 epoch 结束后的打印、作图、checkpoint 保存。

        相比父类，这里额外打印了：
        - `val_ccc`
        - `val_hd95`

        什么时候你会改这里
        ----------
        - 你想改变 best checkpoint 的判据
        - 你想打印更多日志
        - 你想增加额外的可视化产物
        """
        self.logger.log("epoch_end_timestamps", __import__("time").time(), self.current_epoch)

        self.print_to_log_file(
            "train_loss",
            np.round(self.logger.my_fantastic_logging["train_losses"][-1], decimals=4),
        )
        self.print_to_log_file(
            "val_loss",
            np.round(self.logger.my_fantastic_logging["val_losses"][-1], decimals=4),
        )
        self.print_to_log_file(
            "Pseudo dice",
            [
                np.round(i, decimals=4)
                for i in self.logger.my_fantastic_logging["dice_per_class_or_region"][-1]
            ],
        )

        val_ccc = self.logger.my_fantastic_logging["val_ccc"][-1]
        self.print_to_log_file(
            f"val_CCC (volume accuracy): {np.round(val_ccc, decimals=4)}   "
            f"[CCC 接近 1.0 = 体积预测越准确，临床建议 > 0.9]"
        )

        val_hd95 = self.logger.my_fantastic_logging["val_hd95"][-1]
        self.print_to_log_file(
            f"val_HD95 (pseudo boundary distance): {np.round(val_hd95, decimals=4)}   "
            f"[HD95 越接近 0 越好；这里是 patch-level pseudo HD95]"
        )

        self.print_to_log_file(
            "Epoch time: "
            f"{np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s"
        )

        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(
                __import__("os.path", fromlist=["join"]).join(
                    self.output_folder, "checkpoint_latest.pth"
                )
            )

        from batchgenerators.utilities.file_and_folder_operations import join

        if self._best_ema is None or self.logger.my_fantastic_logging["ema_fg_dice"][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging["ema_fg_dice"][-1]
            self.print_to_log_file(
                f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}"
            )
            self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1
