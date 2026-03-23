# nnU-Net v2 深度开发手册：架构、参数与逻辑流

本手册专为需要在 nnU-Net v2 基础上进行模型缝合（如加入 Attention, Transformer, Mamba 等）的开发者编写。它详细解析了从命令行启动到模型运行的每一个核心环节。

---

## 1. `nnUNetv2_train` 核心逻辑全链路 (主逻辑链)

当你执行 `nnUNetv2_train` 时，程序的生命周期如下：

### 1.1 启动与类加载 (`run_training.py`)
- **入口**: `run_training_entry()` 解析命令行。
- **定位 Trainer**: `get_trainer_from_args()` 动态搜索 `nnunetv2/training/nnUNetTrainer` 文件夹，根据 `-tr` 参数加载对应的 Python 类。
- **加载计划 (Plans)**: 通过 `plans.json` 获取数据集的预处理统计信息（如 Patch Size, 步长, 卷积核大小）。

### 1.2 初始化阶段 (`nnUNetTrainer.py -> initialize()`)
- **网络构建**: 
  - 调用 `build_network_architecture()`。
  - 核心跳转到 `nnunetv2/utilities/get_network_from_plans.py` 中的 `get_network_from_plans` 函数。
  - **重要**: v2 版本默认使用 `dynamic_network_architectures` 库来动态生成网络架构，而不是写死的模型文件。
- **损失函数**: 调用 `_build_loss()`。如果是多尺度输出（深度监督），会套用 `DeepSupervisionWrapper`。

### 1.3 训练循环 (`nnUNetTrainer.py -> run_training()`)
1. **`on_train_start()`**: 设置环境变量，初始化增强器。
2. **`train_step()`**: 
   - 送入数据 -> 前向传播 -> 计算 Loss -> `grad_scaler` 缩放 -> `loss.backward()` -> `optimizer.step()`。
3. **`validation_step()`**: 
   - **核心注释**: 在每个 epoch 结束后的验证环中，模型会切换到 `.eval()` 模式，在验证集 patch 上跑一次推理。
   - 这是我们插入 **CCC (体积一致性相关系数)** 计算的位置，因为它最适合在验证阶段监控模型对体积预测的准确度。
4. **`on_validation_epoch_end()`**: 汇总并计算 **Mean Dice**。
5. **`on_epoch_end()`**: 打印日志，判断是否为当前最佳模型 (`checkpoint_best.pth`)。

---

## 2. nnU-Net 原生架构剖析 (不仅仅是 U-Net)

nnU-Net v2 的默认模型叫 **PlainConvUNet**，它是一个高度参数化的、基于对称设计的编码器-解码器架构。

### 2.1 架构特点
- **自适应卷积核与步长**: 它不是固定的 $3 \times 3$。如果你的数据各向异性严重（如 $1.0 \times 1.0 \times 5.0$ mm），它会自动在某些维度使用 $1 \times 3 \times 3$ 的卷积核。
- **深度监督 (Deep Supervision)**: 
  - **训练状态**: 解码器的每一个特征尺度都会通过一个 $1 \times 1$ 卷积产生一个分割输出。
  - **理由**: 辅助弱监督，防止深层梯度消失，强迫每个尺度都学习有意义的分割特征。
- **归一化层**: 默认使用 `InstanceNorm3d`。
- **瓶颈层 (Bottleneck)**: 编码器最底层，特征图尺寸最小，语义信息最浓。这也是缝合 Transformer 或全局 Attention 的最佳位置。

---

## 3. 完整参数说明 (`arch_init_kwargs`)

在 `build_network_architecture` 中，你会接收到一个名为 `arch_init_kwargs` 的字典。其完整内容如下：

| 参数名称 | 类型 | 详细释义 |
| :--- | :--- | :--- |
| `n_stages` | `int` | 总层数（Encoder 下采样的总次数）。通常为 5-6 层。 |
| `features_per_stage` | `list` | 每一层 Stage 的通道数，如 `[32, 64, 128, 256, 512, 512]`。 |
| `kernel_sizes` | `list` | 每一层卷积核的具体尺寸。 |
| `strides` | `list` | 每一层对应的步长。第一个往往是 `[1,1,1]`（不降采样）。 |
| `conv_op` | `class` | 卷积操作类，通常是 `nn.Conv3d`。 |
| `norm_op` | `class` | 归一化操作类，通常是 `nn.InstanceNorm3d`。 |
| `nonlin` | `class` | 激活函数，通常是 `nn.LeakyReLU`。 |
| `nonlin_kwargs` | `dict` | 激活函数的偏置参数，如 `{'inplace': True, 'negative_slope': 0.01}`。 |
| `dropout_op` | `class` | 是否使用 Dropout，默认为 `None`。 |
| `n_conv_per_stage` | `int/list` | Encoder 每一层连续卷积的次数（默认为 2）。 |
| `n_conv_per_stage_decoder` | `int/list` | Decoder 每一层连续卷积的次数（默认为 1）。 |
| `deep_supervision` | `bool` | 是否开启多尺度 Loss 监督。 |

---

## 4. nnU-Net v2 官方推荐的修改方式

根据官方文档 (`documentation/extending_nnunet.md`)，针对不同需求，官方推荐了两种路径：

### 路径 A：快速且有效 (科研首选 - 你目前采用的方式)
- **操作**：实现一个新的 `nnUNetTrainer` 类并重写 `build_network_architecture`。
- **优点**：能够快速验证你的创新点（Attention, Transformer 等），无需深入研究 nnU-Net 复杂的 GPU 显存预估算法。
- **官方提示**：
  - 确保你的架构不带最后的非线性层（Softmax/Sigmoid），nnU-Net 内部处理。
  - 如果不支持多尺度输出，请在 Trainer `__init__` 中设 `self.enable_deep_supervision = False`。

### 路径 B：正式且完整 (工业级/集成级)
- **操作**：编写一套完整的动态架构、一个新的 `ExperimentPlanner` (实验计划器) 和配套的 `Trainer`。
- **复杂度**：极高。需要手写 GPU 显存分配预估代码，确保 `plan_and_preprocess` 能自动算出你的模型。

---

## 5. 最终建议：如何“合法”且“优雅”地修改

1. **不要碰 `nnunetv2/` 核心包的代码**（除非你是在修复官方 Bug）。
2. **永远通过继承来创新**：
   - 所有的模型修改都在你的 `my_archs/` 文件夹下。
   - 所有的训练逻辑修改都在你自定义的 `trainer_attention.py` 中。
3. **“合法”渠道的意义**：这样当 nnU-Net 官方更新版本（修复 Bug 或提升推理速度）时，你只需要用 `git pull` 更新核心包，而不必担心你自己辛辛苦苦缝合的代码被覆盖或导致冲突。

**你可以把这套 Trainer + 自定义模型看作是 nnU-Net 官方留给你的“合法插件接口”。** 
