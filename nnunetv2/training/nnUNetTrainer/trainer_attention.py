from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.ccc_metric import compute_ccc
from nnunetv2.utilities.hd95_metric import compute_hd95




# MyTrainer_Attention 你可以改成你想要的名字，最后记得训练或者啥的时候，带上-tr XXXX（新名字）

class MyTrainer_Attention(nnUNetTrainer):

    """
    这是一个nnUNetTrainer的子类。
    我来介绍我们现在要做的事情。你听说过“缝合模块”吧，就是那种你不改原来网络结构，直接在某一层的输出上缝合一个新的模块，。

    nnUNet的特点是它是根据你的数据集创造一个你专有的模型,但是这个模型是隐形的,你看不见,不像很简单的代码里,有一个model.py
    所以我做了一个inspect_plans_models.py,你在这里面可以根据数据预处理的结果,看出nnUNet给你的模型长啥样,你就可以针对性的改.

    MyTrainer_Attention是一个训练器,所谓训练器,意思很简单,就是有模型架构,loss啊,优化器啊,训练步骤...每个方法我都会给你写好注释.
    """

    # 这个魔法方法就是初始化参数，你不需要动就行啦
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
            nnU-Net 规划器生成的总配置字典。（这个配置你别动）
            一般对于我们脂肪分割的数据不变，plans每次大概率还是那个东西
            它来自 `nnUNet_preprocessed/<DatasetName>/<plans_identifier>.json`。

            这里面包含的不是“单纯网络参数”，而是整套实验规划信息，例如：
            - 不同 configuration 的定义
            - patch size
            - batch size
            - network class name
            - network init kwargs
            - spacing / resampling / cascade 相关信息

            父类会把它包装成 `PlansManager`，后续很多接口都从这里读配置。
        —---------
        configuration:（一般为了良好的性能，我们选择 `3d_fullres` 这个配置，毕竟都比较了，肯定挑最好的）
            当前训练配置名，通常是：
            - `2d`
            - `3d_fullres`
            - `3d_lowres`
            - `3d_cascade_fullres`

            它会决定当前训练到底选用哪套 patch size、batch size、network kwargs。
        -----------
        fold:（也就是传说中的五折验证）
            当前交叉验证 fold。
            常见取值是 `0~4`，也可能是 `all`。

            这个参数最终会影响：
            - 数据划分
            - 输出目录
            - checkpoint 位置
        -----------
        dataset_json:（这个你会碰到的，自己写的，里面就告诉你了一些基本信息）
            预处理目录中的 `dataset.json` 内容。
            它会告诉 nnU-Net:
            - 谁是背景？
            - 是不是CT?还是磁共振?
            - 有多少数据？
            基本就这些信息
        -----------
        device:（这个暂时无视掉）
            当前训练设备。
            常见是：
            - `torch.device("cuda")`——如果你有 NVIDIA GPU,通常就是这个了。
            - `torch.device("cpu")`——如果你没有 GPU,或者想在 CPU 上测试代码。
            - `torch.device("mps")`——如果你有 Apple M1/M2/M3/M4/M5芯片,可以使用这个。

            父类还会根据 DDP 情况重新修正实际使用的 device。
        """
        super().__init__(plans, configuration, fold, dataset_json, device)

        # 有一个叫做enable_deep_supervision的成员变量，默认是True，也就是深监督，我并不推荐开启,如果你想开启，设置一下就行
        self.enable_deep_supervision = False

    # 改优化器和学习率调度器
    # 也就是这里也是一个重点
    def configure_optimizers(self):
        """
        默认返回的是super(self,self).configure_optimizers()，也就是父类 nnUNetTrainer 的优化器配置。
        这里解释一下,super(self,self)其实就是MRO调用链
        找nnUNetTrainer里面的configure_optimizers函数,直接拿来用就行了。
        ----------
        
        ----------
        这里默认会返回一个元祖,(optimizer,scheduler)其中：
        - optimizer 优化器,也就是你天天看到的什么Adam啊,SGD啊
        - scheduler 是一个学习率调度器对象,比如PolyLR,就是随着epoch增加,学习率逐渐降低的意思。因为很少我们学习率会是固定不变的
        """

        # 要改的话很简单，右键configure_optimizers，转到定义
        return super().configure_optimizers()

    # 这东西别动，一般来说我们不需要动它
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

    # 这玩意建议直接关掉
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
        # 这个就是关掉啦，有的函数不需要return，只需要单纯修改一下状态就可以了。
        self.enable_deep_supervision = False


    # 修改loss在这里！！！你的损失函数，创新点，你除了模块，损失函数也是一个很有意思的东西
    # 也就是说，_build_loss是决定用什么损失函数的地方。
    def _build_loss(self):
        """
        这个接口负责构造训练使用的 loss。

        它没有显式函数参数，但会依赖 trainer 当前状态中的这些成员：
        - `self.label_manager`
        - `self.configuration_manager.batch_dice`
        - `self.label_manager.ignore_label`
        - `self.is_ddp`
        - `self.enable_deep_supervision`

        
        """
        from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

        # region-based segmentation任务是指，有的奇葩分类任务，可能一个像素点可能同时属于多个类别
        # （比如脂肪分割里，可能既有“内脏脂肪”又有“皮下脂肪”），但是我们脂肪分割这个就是简单的二分类，不用担心
        # 我为了保持原来的代码结构，还留了这个判断哈
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

        # 看这里！！！！这才是我们用的
        # 如果你要用什么很牛逼的损失函数，可以跳转到compound_losses定义，然后在这里加一个你想要的loss
        # 推荐损失函数用AI写，AI写这个写得很准很准！你把你想要的东西，跟AI介绍详细，就行了，然后import过来
        else:
            from nnunetv2.training.loss.compound_losses import DC_and_CE_loss

            # 这个是Dice + 交叉熵混合的loss，很常用的
            # 怎么改呢？做法很简单，右键compound_losses转到定义，然后进行这个文件，新建一个loss类（让AI写，描述好你的想法）
            # 之后在这里把DC_and_CE_loss改成你新建的那个类就行了，记得传入正确的参数哦。
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

    # 这里虽然参数比较多，你可以选择看一下参数的函数，其实不看也没事，因为我已经将网络显性构建了，并且我也做了一个例子模块，你可以直接在我那里改
    # 也就是问题我们转化成了”搭积木“问题。
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
        ----------不重要
        architecture_class_name:
            plans 里记录的原始网络类名。
            在原版 nnU-Net 里，这个参数会决定默认架构如何被构建。

            但在你当前这个自定义 trainer 中，我们并不直接依赖它来实例化网络，
            因为我们要手动 `return UNetARTBlock(...)`。

            它保留在函数签名里，主要是为了兼容 nnU-Net 的统一调用方式。

        -----------重要！！
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

        -----------不重要
        arch_init_kwargs_req_import:
            某些 kwargs 对应的类或函数需要动态导入时，nnU-Net 会用到这个字段。

            对你当前这个手写网络来说，它通常不是最核心的参数，
            但函数签名必须保留，才能和父类的调用规范对齐。

        ------------你觉得这个重要不重要？
        num_input_channels:
            网络输入通道数。
            它不是你手工写死的，而是 nnU-Net 根据数据集自动推出来的。

            例如：
            - 单模态 CT/MR 可能是 `1`
            - 多模态输入可能是 `2`、`4` 等

            你自定义网络第一层一定要和它对齐。

        ------------我猜很重要
        num_output_channels:
            网络输出通道数，也就是 segmentation heads 数量。
            这个值由 `label_manager` 决定，不一定等于“类别总数”本身。

            原因是 nnU-Net 支持：
            - 普通 class-based segmentation
            - region-based segmentation
            - ignore label 这个意思就是不参加训练的标签

            所以最稳的做法就是永远直接使用这个参数，不要自己手写类别数。

        -------------我已经帮你关了
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


        # 这个print很重要，比如你输入指令-tr MyTrainer_Attention，
    
        # 但是怎么知道是不是真调用了呢？就看这个print，如果有，说明数据走到这里了！
        print("MyTrainer_Attention build_network_architecture called")
        print("\nnum_input_channels:", num_input_channels)
        print("\nnum_output_channels:", num_output_channels)
        print("\nenable_deep_supervision from parent:", enable_deep_supervision) # 这个设置成False，最好别用深层次监督，不然不是很好写loss
        print("\nignored architecture_class_name:", architecture_class_name)
        print("\narch_init_kwargs keys:", list(arch_init_kwargs.keys())) # 拿值
        print("\narch_init_kwargs_req_import:", arch_init_kwargs_req_import)

        #———————————————————————————————看这里！！！！————————————————————————————————#
       

        from nnunetv2.training.my_archs.Net import YourNet

        base_net = nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=False,  # 这里直接关掉深监督了，毕竟我们这个网络不太适合多尺度输出，不关闭的话，就把=False改成enable_deep_supervision
        )

        # 对着YourNet右键，“跳转到定义”，跳转到我们的YourNet那里，里面我写了具体怎么进行修改模块。
        Net = YourNet(base_net=base_net)
        return Net


    # ————————————————————————————底下别动了，指标我都基本帮你写好了————————————————————————————————#
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
