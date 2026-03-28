# 如何修改 nnU-Netv2 的模型架构

这份教程只讲你现在项目里这一条实际可用的路线：

- 训练入口：`nnunetv2/training/nnUNetTrainer/trainer_attention.py`
- 网络包装：`nnunetv2/training/my_archs/Net.py`
- 可插拔模块：`nnunetv2/training/my_archs/MyBlock.py`

一句话概括：

> 先让 nnU-Net 按 `plans.json + dataset.json` 搭出默认网络，再用 `YourNet` 去替换其中某一层。

---

## 1. 训练时到底发生了什么？

训练命令示例：

```bash
nnUNetv2_train DATASET_ID 3d_fullres 0 -tr MyTrainer_Attention
```

这条命令的核心流程是：

```text
nnUNetv2_train
-> run_training.py
-> 找到 MyTrainer_Attention
-> 读取 plans.json 和 dataset.json
-> MyTrainer_Attention.build_network_architecture(...)
-> 先构造默认 nnU-Net 网络 base_net
-> 再把 base_net 传给 YourNet(base_net)
-> YourNet 里替换某一层
-> 训练时真正执行 self.network(data)
```

所以你现在这套代码，不是“整网重写”，而是：

- trainer 负责拿到默认网络
- `YourNet` 负责对默认网络做局部修改

---

## 2. 你现在主要改哪几个文件？

### A. `trainer_attention.py`

作用：

- 注册你的 trainer
- 决定 loss
- 决定 optimizer / lr scheduler
- 在 `build_network_architecture(...)` 里先拿到默认网络，再交给 `YourNet`

当前项目里的核心逻辑就是：

```python
base_net = nnUNetTrainer.build_network_architecture(...)
net = YourNet(base_net=base_net)
return net
```

### B. `Net.py`

作用：

- 接收 `base_net`
- 找到你想替换的层
- 用 `WrappedStage(...)` 替换掉
- `forward` 里继续走 `self.base_net(x)`

也就是说，你真正“改层”的地方就在这里。

### C. `MyBlock.py`

作用：

- 写你自己的模块
- 写一个包装层 `WrappedStage`
- 让数据流变成：

```text
x -> 原来的 old_stage -> 你的 MyBlock -> out
```

这就是最简单的“插拔模块”。

---

## 3. 现在这套模板是怎么改层的？

你现在的 `Net.py` 里，演示了两种替换：

- `self.base_net.encoder.stages[2]`
- `self.base_net.encoder.stages[-1]`

其中：

- `encoder.stages[2]` 是中间某一层 encoder
- `encoder.stages[-1]` 是 bottleneck

注意：

> 在你现在这个默认 `PlainConvUNet` 里，bottleneck 也算在 encoder 里面，所以要用 `encoder.stages[-1]` 去取，而不是写成 `self.base_net.bottleneck`。

---

## 4. 新手改层时最重要的原则

你第一次改模块时，强烈建议守住这条：

- 输入 shape 不变
- 输出 shape 不变
- 通道数不变

比如原来这一层输出是 `128` 通道，那你自己的模块最好也输出 `128` 通道。

这样最稳，不容易把后面的 decoder 和 loss 一起改炸。

---

## 5. 怎么决定该改哪一层？

先运行：

```bash
python inspect_plans_model.py 你的nnUNetPlans.json路径 -dj 你的dataset.json路径 -c 3d_fullres
```

然后看生成的 `model_summary.md`。

你要重点看：

- `Encoder Stage 0/1/2/...`
- `Bottleneck`
- `Decoder Stage ...`

一般最适合插模块的是：

- `encoder.stages[2]` 或 `encoder.stages[3]`
- `encoder.stages[-1]` 也就是 bottleneck

---

## 6. 什么时候改 trainer，什么时候改 network？

### 改 `trainer_attention.py`

适合：

- 改 loss
- 改 optimizer
- 改学习率调度
- 改 deep supervision 开关

### 改 `Net.py` / `MyBlock.py`

适合：

- 改某一层网络
- 加 attention
- 加 MLP
- 加门控
- 做特征融合

一句话：

> 训练流程改 trainer，模型计算图改 network。

---

## 7. 你现在应该怎么学

推荐顺序：

1. 先看 `trainer_attention.py`，理解 `base_net -> YourNet` 这条链
2. 再看 `inspect_plans_model.py`，先知道默认网络长什么样
3. 再看 `Net.py`，理解“替换层”到底是在替换什么
4. 最后看 `MyBlock.py`，把你自己的模块写进去

这就是你当前项目里最实用的一条路线。
