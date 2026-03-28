# nnU-Net 自定义修改简表

这份文档是短版说明，只对应你当前项目的真实实现。

---

## 1. 当前项目怎么改网络？

当前路线不是自己从零写一个完整网络，而是：

1. 用 nnU-Net 默认逻辑，根据 `plans.json + dataset.json` 构造一个默认网络 `base_net`
2. 把 `base_net` 传给 `YourNet(base_net)`
3. 在 `YourNet` 里替换某一层

所以你现在的核心思路是：

```text
默认网络 -> 包装 -> 局部替换
```

---

## 2. 三个关键文件

### `trainer_attention.py`

负责：

- 自定义 trainer
- `_build_loss()`
- `configure_optimizers()`
- `build_network_architecture(...)`

这里最重要的是：

```python
base_net = nnUNetTrainer.build_network_architecture(...)
net = YourNet(base_net=base_net)
return net
```

### `Net.py`

负责：

- 接收 `base_net`
- 找到要改的层
- 用 `WrappedStage(...)` 替换

### `MyBlock.py`

负责：

- 写你自己的小模块 `MyBlock`
- 写包装层 `WrappedStage`

---

## 3. 现在模板里已经演示了什么？

当前模板演示了两种替换：

- `self.base_net.encoder.stages[2]`
- `self.base_net.encoder.stages[-1]`

其中：

- `stages[2]`：中间 encoder 层
- `stages[-1]`：bottleneck

注意：

> 你现在这套默认 `PlainConvUNet` 里，bottleneck 算在 encoder 里面，所以要从 `encoder.stages[-1]` 取。

---

## 4. 最重要的规则

第一次改模块时，先尽量保持：

- 输入 shape 不变
- 输出 shape 不变
- 通道数不变

这样最适合新手，也最不容易把后面的网络接坏。

---

## 5. 怎么看默认网络长什么样？

用这个脚本：

```bash
python inspect_plans_model.py 你的nnUNetPlans.json路径 -dj 你的dataset.json路径 -c 3d_fullres
```

重点看输出的：

- `model_summary.md`

你会看到：

- 默认网络骨架
- 每一层的维度变化
- 适合改哪一层

---

## 6. 当前模板更适合谁？

这套模板适合：

- 第一次接触 nnU-Net 二次开发的人
- 只会搭积木式改模块的人
- 想做局部替换，而不是整网重写的人

如果你只是想：

- 在某一层后面加 attention
- 在 bottleneck 加模块
- 在中间层做小改动

那这套模板就够用了。

---

## 7. 当前版本的限制

当前 `MyBlock.py` 用的是 `Conv3d`，所以这套示例默认是面向：

- `3d_fullres`
- `3d_lowres`

如果你要改 `2d` 配置，需要把里面的 `Conv3d` 改成 `Conv2d`，或者自己做 2D/3D 兼容。

---

## 8. 一句话总结

当前项目的推荐路线就是：

> 先看默认网络，再在 `Net.py` 里替换某一层，最后只在 `MyBlock.py` 里写自己的模块。
