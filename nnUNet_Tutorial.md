# 如何修改nnU-Netv2的模型架构呢？

我们要做创新，最好做两件事情：

- 对模型某个模块进行创新，不是让你加入ViT，或者加入Mamba就是创新。
- loss函数很适合作为创新，比如它用来限制什么，但是这个是可选项。

也就是说，你真正要改的重点不是 dataloader、preprocess，而是：

- `nnunetv2/training/nnUNetTrainer/trainer_attention.py` ——> 修改训练器
- `nnunetv2/training/my_archs/Net.py` ——> 修改真正的模型，我给了一个很小很小的模型，你在上面改就行了。



## 1. 当你执行 `nnUNetv2_train ... -tr MyTrainer_Attention` 时，函数链到底怎么走？

训练命令示例：

```bash
nnUNetv2_train DATASET_ID 3d_fullres 0 -tr MyTrainer_Attention
```
稍微解释一下这个指令的意思。
> 我执行训练指令，dataset的名字，执行的配置是3d_fullres,fold选择只做第0个，然后选择的模型是自己创造的MyTrainer_Attention。


### 1.1 我们输入指令，这个模型到底怎么运行的？

```text
当你执行nnUNetv2_train这条指令的时候，你可以看到pyproject.toml文件中写了具体的调用链条。
```

### 1.2 对应源码位置

#### 第 1 步：命令行入口

- 文件：`nnunetv2/run/run_training.py`
- 函数：`run_training_entry()`
- 作用：解析 `dataset_id / configuration / fold / -tr / -p / --val` 等命令行参数。

#### 第 2 步：进入训练总调度

- 文件：`nnunetv2/run/run_training.py`
- 函数：`run_training(...)`
- 作用：决定单卡还是 DDP，决定是否继续训练、是否只验证、是否加载预训练权重。

#### 第 3 步：根据 `-tr MyTrainer_Attention` 找到你的 trainer

- 文件：`nnunetv2/run/run_training.py`
- 函数：`get_trainer_from_args(...)`
- 关键动作：
  - 调用 `recursive_find_python_class(...)`
  - 在 `nnunetv2/training/nnUNetTrainer/` 目录下查找名字严格匹配的类
  - 所以你命令里写的是 `MyTrainer_Attention`，类名也必须就是 `MyTrainer_Attention`

#### 第 4 步：加载 plans 和 dataset.json

- 同样在 `get_trainer_from_args(...)` 中完成
- 会读取：
  - `nnUNet_preprocessed/<DatasetName>/<plans_identifier>.json`
  - `nnUNet_preprocessed/<DatasetName>/dataset.json`

这些信息会被传给你的 trainer：

```python
MyTrainer_Attention(
    plans=plans,
    configuration=configuration,
    fold=fold,
    dataset_json=dataset_json,
    device=device,
)
```

#### 第 5 步：实例化 trainer，但这时网络还没真正建出来

实例化 `MyTrainer_Attention` 时，主要是在保存配置、路径、训练超参数、label manager 等状态。

真正把网络构造出来，是在后面的：

```python
initialize()
```

#### 第 6 步：`initialize()` 才是模型初始化核心

文件：`nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`

调用顺序大致是：

```text
initialize()
-> _set_batch_size_and_oversample()
-> determine_num_input_channels(...)
-> build_network_architecture(...)
-> configure_optimizers()
-> _build_loss()
-> infer_dataset_class(...)
```

这里最重要的两个“可插拔接口”就是：

- `build_network_architecture(...)`
- `_build_loss()`

也就是说：

- 你要换模型结构，主要改 `build_network_architecture(...)`
- 你要换 loss，主要改 `_build_loss()`

#### 第 7 步：`on_train_start()` 才去构建 dataloader

调用顺序：

```text
run_training()
-> on_train_start()
-> get_dataloaders()
-> set_deep_supervision_enabled(...)
```

这一步说明一个很重要的事实：

> nnU-Net 的数据处理链路和模型链路是分开的。你完全可以只改 trainer 和 network，而不碰 dataloader / preprocess。

#### 第 8 步：真正的模型 forward 在哪？

真正执行模型前向传播的地方，不在 `build_network_architecture(...)`，而是在：

```python
train_step()
```

和

```python
validation_step()
```

里面这句：

```python
output = self.network(data)
```

所以要分清两件事：

- `build_network_architecture(...)`：负责“创建模型对象”
- `self.network(data)`：负责“执行模型前向传播”

#### 第 9 步：epoch 循环

训练主循环在：

- 文件：`nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`
- 函数：`run_training()`

流程是：

```text
for epoch in range(...):
    on_epoch_start()
    on_train_epoch_start()
    多次 train_step()
    on_train_epoch_end()
    on_validation_epoch_start()
    多次 validation_step()
    on_validation_epoch_end()
    on_epoch_end()
```

#### 第 10 步：训练结束后的真实验证

训练完成后还会跑：

```python
perform_actual_validation()
```

这一步不是 patch 级的 online validation，而是更完整的滑窗推理导出和评估。

---

## 2. 你以后要改模块，应该改哪一层？

这个问题非常关键。

### 2.1 不建议改的数据层

这些通常别动：

- `nnUNetv2_extract_fingerprint`
- `nnUNetv2_plan_experiment`
- `nnUNetv2_preprocess`
- `nnUNetTrainer.get_dataloaders()`
- `nnunetv2/training/dataloading/*`

因为这些是 nnU-Net 最值钱、最稳定的自动化资产。

### 2.2 建议改的模型层

你真正应该改的是这两层：

#### A. trainer 层

文件：

- `nnunetv2/training/nnUNetTrainer/trainer_attention.py`

职责：

- 注册你的自定义 trainer
- 决定构造哪一个网络
- 决定 loss
- 决定 deep supervision 开关
- 决定 optimizer / lr scheduler
- 决定是否记录额外指标

#### B. network 层

文件：

- `nnunetv2/training/my_archs/unet_art_block.py`

职责：

- 真正定义网络结构
- 真正往模型里加 `nn.Linear`、Attention、MLP、门控、分支、融合模块

一句话总结：

> trainer 决定“训练系统怎么组织”，network 决定“模型到底长什么样”。

---

## 3. `MyTrainer_Attention` 里最值得你改的接口

### 3.1 `__init__(...)`

用途：

- 接收 plans、configuration、fold、dataset_json、device
- 设置训练级超参数
- 可以在这里修改一些 trainer 级别默认行为

典型适合改：

- `self.enable_deep_supervision`
- `self.initial_lr`
- `self.weight_decay`
- `self.num_epochs`

### 3.2 `build_network_architecture(...)`

这是最重要的接口。

用途：

- 根据 nnU-Net 提供的配置参数，返回一个真正的 `nn.Module`

你在这里做的事情通常是：

- 不再使用默认网络
- 直接 `return UNetARTBlock(...)`
- 或者以后 `return YourOwnNet(...)`

注意：

- 这里是“造网络对象”
- 不是“执行 forward”

### 3.3 `_build_loss()`

用途：

- 决定训练时到底用哪个 loss

如果你的网络不输出多尺度结果：

- 建议关闭 deep supervision
- 并返回普通的 `DC_and_CE_loss`
- 不要再套 `DeepSupervisionWrapper`

### 3.4 `set_deep_supervision_enabled(...)`

用途：

- 父类默认会尝试去改网络 decoder 上的 deep supervision 开关

如果你的自定义网络没有 `decoder.deep_supervision` 这种结构：

- 最安全的办法就是像现在这样重写它
- 直接强制 `self.enable_deep_supervision = False`

### 3.5 `configure_optimizers()`

用途：

- 决定 optimizer 和 lr scheduler

以后如果你想改：

- SGD -> Adam / AdamW
- PolyLR -> CosineAnnealing / Warmup

通常就在这个接口里处理。

### 3.6 `train_step(...)`

用途：

- 执行一次训练迭代
- 真正调用 `self.network(data)`
- 真正执行 backward

如果你只是改模型结构，一般不用改这里。

### 3.7 `validation_step(...)`

用途：

- 执行一次验证迭代
- 做 online metric 统计

如果你要额外记录体积指标、分类指标、边界指标，可以重写这里。

### 3.8 `on_validation_epoch_end(...)`

用途：

- 聚合整个 epoch 的验证结果
- 记录到 logger

你现在的 CCC 指标就是在这里做 epoch 级汇总的。

---

## 4. 什么时候该把 `nn.Linear` 写进 trainer，什么时候不该？

结论先说：

> `nn.Linear`、Attention block、MLP、特征融合层，这些都应该放在 network 文件里，而不是直接塞进 trainer 里。

原因：

- trainer 负责训练流程
- network 负责模型计算图
- `self.network(data)` 调的是 network 的 `forward(...)`

所以推荐工作流是：

1. 在 `unet_art_block.py` 里加你的模块。
2. 在 `trainer_attention.py` 的 `build_network_architecture(...)` 中返回这个新网络。
3. 如果输出形式变了，再同步修改 `_build_loss()` 和 `set_deep_supervision_enabled(...)`。

---

## 5. 数据预处理链为什么可以完全不动？

因为 nnU-Net 已经帮你做掉了最复杂、最不想自己维护的那些事情：

- 数据指纹提取
- spacing 自适应
- patch size 规划
- batch size 规划
- 强度归一化
- 前景裁剪
- 重采样
- 训练时 patch 采样
- 数据增强


所以你的最佳策略是：

- 预处理层别碰
- dataloader 层别碰
- trainer 和 network 层精改

---

## 6. 你应该做的

1. 找到`nnunetv2.run.training.nnUNetTrainer.trainer_attention.py。
2. 我在里面写了很详细怎么改，你看到就知道大概怎么改啦。

---

## 7. 推荐你接下来的修改顺序

1. 先看 `trainer_attention.py`，把接口意义彻底搞懂。
2. 再看 `unet_art_block.py`，把你要加的模块放到真正的网络里。

这套路线最稳，也最符合 nnU-Net 的设计哲学。
