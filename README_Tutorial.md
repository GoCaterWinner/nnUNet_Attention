# nnU-Net 自定义模块添加 & 数据处理教程

本篇 README 是针对 nnU-Net 框架的保姆级教程，结合本仓库中的 `MyTrainer_Attention` 示例，详细拆解了 nnU-Net 如何添加自定义网络模块，以及它背后的复杂数据预处理逻辑和训练运行逻辑。

---

## 1. nnU-Net 训练运行逻辑链路

当你敲下这行代码开始训练时：
```bash
nnUNetv2_train DATASET_ID 2d 0 -tr MyTrainer_Attention
```

背后究竟发生了什么？请看下方的思维执行流程图：

```mermaid
graph TD
    A[输入: nnUNetv2_train -tr MyTrainer_Attention] --> B[1. run_training_entry 解析命令行参数]
    B --> C[2. 加载 dataset.json 和 plans.json]
    C --> D[3. 通过 Python 反射机制在 Trainer 目录下查找 \nMyTrainer_Attention 类并实例化]
    D --> E[4. 执行 trainer.initialize\(\)]
    E --> F[5. 调用 _build_loss\(\) \n配置损失函数]
    E --> G[6. 调用 build_network_architecture\(\) \n注入你的 UNetARTBlock]
    E --> H[7. 基于 plans 加载并配置 DataLoader]
    F --> I[8. 执行 trainer.run_training\(\) 进入 Epoch 循环]
    G --> I
    H -->|输出: Batch数据 \n Image: (B, C, H, W) 或 (B, C, D, H, W) | I
    I --> J[前向传播: 自定义网络模块 UNetARTBlock \n输入: DataLoader 输出的 Image Tensor \n输出: 预测的 Logits 例如 (B, Class, H, W)]
    J --> K[Loss 计算模块 \n输入: Logits 和 Label 标注 \n输出: Loss 标量值 (Scalar)]
    K --> L[反向传播 Backprop -> Checkpoint 保存]
```

核心结论：
**所有的自定义修改，都可以通过继承 `nnUNetTrainer` 并重写对应的函数（如 `_build_loss` 和 `build_network_architecture`）来实现。**

---

## 2. 如何在 `MyTrainer_Attention` 里面修改/添加模块？

### A. 架构替换核心点
如果想要将原版的 nnU-Net 替换为含 Attention 或其他定制结构的模块，你需要在 `MyTrainer_Attention` 类中覆盖（Override）`build_network_architecture` 方法：

- **实现机制**：在此函数内不调用 `super()` 的原本网络，而是直接 `return` 你的网络（见 `trainer_attention.py` 中的 `UNetARTBlock` 实例化）。
- **必须接收的参数**：
  - `num_input_channels`：由任务模态数决定。
  - `num_output_channels`：由目标类别数决定。

### B. 重点需要注意的超参数 (Hyperparameters)
在你客制化模块时，以下超参数极易导致报错，请格外留心：

1. **`enable_deep_supervision` (深度监督开关)**
   - **默认行为**：nnU-Net 严格依赖深度监督（返回 `List[Tensor]` 而不是单一 `Tensor`），默认开启。
   - **修改建议**：如果你的 Attention 分支没有设计多层次的输出，**必须在 `__init__` 中将 `self.enable_deep_supervision = False`**。同时建议重写 `set_deep_supervision_enabled` 强制返回 `False` 拦截父类修改。
2. **`Loss Function` (损失函数组合)**
   - 单层输出如果走进了带有深度监督包装器的 Loss 会导致维度报错。
   - **修改建议**：重写 `_build_loss` 函数，手动返回 `DC_and_CE_loss`（Dice + Cross Entropy），不套 `DeepSupervisionWrapper`。
   - **权重分配**：你可以在 `DC_and_CE_loss` 中自由调整 `weight_ce` 和 `weight_dice`。
3. **`batch_dice`**
   - 控制计算 Batch 层面的 Dice 还是单张 Image 层面的 Dice。默认开启 `batch_dice` 可以让小目标的训练更稳定。

### C. 网络输入输出形状
- **输入维度**：nnU-Net 的 DataLoader 给的 `x`，在 2D 任务中是 `(B, C, H, W)`，在 3D 任务中是 `(B, C, D, H, W)`，务必确保你的 Attention 模块适配对应维度（可通过 kwargs 传入的 plan 信息动态构建 2D/3D Conv 算子）。
- **输出维度**：最终预测应当匹配 `(B, num_output_channels, 相关空间维度)`，且 **不需要** 在末尾加 Softmax/Sigmoid，Loss 内部或预测时会自动处理 logits。

---

## 3. nnU-Net 复杂的数据预处理流程解析

nnU-Net 强大霸道的地方在于它“自适应”的预处理。从你按标准格式准备好 `raw` 数据，到可以训练，它完整执行了三步：

```mermaid
graph TD
    A[输入: nnUNet_raw 原始 NIfTI/PNG 图像和标签 \n (Image: 各种未经处理的异常尺寸与物理间距)] --> B[1. nnUNetv2_extract_fingerprint]
    B -->|输出: dataset_fingerprint.json \n(包含 Spacing, 灰度均值, 极值等物理指纹)| C[2. nnUNetv2_plan_experiment]
    C -->|输出: nnUNetPlans.json \n(包含目标 Patch Size, Batch Size 等极苛刻的最佳配置)| D[3. nnUNetv2_preprocess \n根据计划执行并行处理]
    D --> E[Resampling: 统一物理层面的 Target Spacing]
    E --> F[Cropping: 裁剪图像全零/无用背景区]
    F --> G[Normalization: 对 CT 做 Z-Score, 对 MRI 做 MinMax 等]
    G -->|输出: 高度对齐的 Numpy 张量数据 .npz \n与相关元属性文件 .pkl| H[存入 nnUNet_preprocessed 对应架构文件夹]
    H --> I[nnUNetv2_train DataLoader \n读取高密度 .npz 从而跳过 I/O 瓶颈]
```

### 三大步骤详情：
1. **提取指纹 (Extract Fingerprint)**:
   - 系统读取你的所有原始图像，计算前景（非0区）的强度属性（Mean, Std, 0.5和99.5的百分位数）。如果是 CT 图像，它会自动识别并执行专门的强度裁剪。这决定了后期 Normalize 的参数。
2. **实验规划 (Plan Experiment)**:
   - 全篇最精彩的启发式规则（Heuristic Rules）所在。
   - 算法会根据中位数 Spacing 设定目标分辨率。
   - 然后计算 VRAM（显存）开销。计算当前给定网络构架在对应 Patch Size 需要的显存条目。若超出 8GB/12GB 限额，则先减缩 Batch Size，再减缩 Patch Size。如果仍然过载或图像极大，就会额外生成 `3d_lowres` 的级联(Cascade) Plans。
3. **极速预处理 (Preprocess)**:
   - 这是非常耗时的阶段。nnU-Net 使用三阶样条插值 (Spline 3rd order) 重采样图像，把形状和物理尺寸对齐，通过 Z-score 标准化强度（使用步骤1得出的 Mean 和 Std），并将结果压缩保存为 `.npz` 以突破训练时的 I/O 瓶颈。

---

> 🎉 **总结:** 在了解以上链路后，你现在可以从容地在 `MyTrainer_Attention` 这个壳子里开发你自创的网络，享受 nnU-Net 带来的数据处理红利和强大骨架！
