# Usage Guide

这份说明是给第一次接触 nnU-Net 的同学准备的。

默认场景：

- 你已经拿到了老师提供的模型文件
- 你主要想做的是“用现成权重进行预测”
- 你不需要自己重新训练模型
- 你可能只会使用 PyCharm

如果你只拿到一个单独的 `checkpoint_best.pth` 或 `checkpoint_final.pth` 文件，请先不要开始。
单独一个 `.pth` 文件通常不能直接用于预测。老师应当提供的是“完整模型文件夹”或者“打包好的 zip 文件”。


**一、你会拿到什么**

最理想的情况，是老师给你一个模型包，解压后大致长这样：

```text
My_Model_Package/
├── dataset.json
├── plans.json
├── fold_0/
│   ├── checkpoint_best.pth
│   ├── checkpoint_final.pth
│   └── validation/
│       └── summary.json
├── fold_1/
├── fold_2/
├── fold_3/
└── fold_4/
```

最少也应该有：

- `dataset.json`
- `plans.json`
- `fold_0/` 这样的文件夹
- `checkpoint_best.pth` 或 `checkpoint_final.pth`

如果只有一个 `.pth` 文件，没有上面这些内容，请联系老师补齐模型包。


**二、怎么打开命令行**

如果你使用 PyCharm：

1. 打开这个项目
2. 点击下方的 `Terminal`
3. 确认当前路径在项目根目录

项目根目录大概像这样：

```text
.../nnUNet_Attention
```


**三、预测前要准备什么**

你需要准备 3 个文件夹：

- `输入文件夹`：放待预测图像
- `输出文件夹`：保存预测结果
- `模型文件夹`：老师给你的模型包解压后的文件夹

例子：

```text
/Users/your_name/Desktop/input_cases
/Users/your_name/Desktop/predictions
/Users/your_name/Desktop/My_Model_Package
```

注意输入图像命名：

- 单模态图像通常需要以 `_0000` 结尾
- 例如：`case001_0000.nii.gz`
- 文件后缀必须和训练时一致，常见是 `.nii.gz`


**四、最常用的预测命令**

推荐使用下面这个命令，因为它不要求你额外配置 `nnUNet_results` 环境变量。

如果你的电脑只有 CPU：

```bash
nnUNetv2_predict_from_modelfolder \
  -i "/你的输入文件夹" \
  -o "/你的输出文件夹" \
  -m "/老师给你的模型文件夹" \
  -f 0 \
  -chk checkpoint_best.pth \
  -device cpu
```

如果你的电脑有可用 GPU：

```bash
nnUNetv2_predict_from_modelfolder \
  -i "/你的输入文件夹" \
  -o "/你的输出文件夹" \
  -m "/老师给你的模型文件夹" \
  -f 0 \
  -chk checkpoint_best.pth \
  -device cuda
```

说明：

- `-i`：待预测图像所在文件夹
- `-o`：预测结果输出文件夹
- `-m`：模型文件夹，注意是“包含 `fold_0`、`fold_1` ... 的上一级文件夹”
- `-f 0`：表示使用 `fold_0`
- `-chk checkpoint_best.pth`：表示使用最优权重
- `-device cpu`：用 CPU 跑，速度较慢但最稳

如果老师明确告诉你模型有完整 5 折，并且希望你使用全部折一起预测，可以把 `-f 0` 改成：

```bash
-f all
```

更完整的例子：

```bash
nnUNetv2_predict_from_modelfolder \
  -i "/Users/your_name/Desktop/input_cases" \
  -o "/Users/your_name/Desktop/predictions" \
  -m "/Users/your_name/Desktop/My_Model_Package" \
  -f 0 \
  -chk checkpoint_best.pth \
  -device cpu
```


**五、预测结果会保存在哪里**

预测完成后，分割结果会出现在你指定的输出文件夹里，比如：

```text
/Users/your_name/Desktop/predictions
```

输出文件的名字通常和输入文件对应，只是内容变成了预测分割结果。


**六、怎么查看验证集 CCC**

CCC 是体积一致性相关系数。一般来说：

- 越接近 `1` 越好
- 常见取值范围是 `-1` 到 `1`

最简单的方法，是直接看模型文件夹里的：

```text
fold_0/validation/summary.json
```

如果你用 PyCharm：

1. 打开 `fold_0/validation/summary.json`
2. 搜索 `CCC`

你会看到类似：

```json
"CCC": 0.93
```


**七、用命令查看 CCC**

如果你想在 Terminal 里查看，可以用下面这些命令。

先把 `summary.json` 整理得更容易看：

```bash
python -m json.tool "/老师给你的模型文件夹/fold_0/validation/summary.json"
```

如果你只想快速搜索 `CCC`：

```bash
grep -n '"CCC"' "/老师给你的模型文件夹/fold_0/validation/summary.json"
```

例如：

```bash
python -m json.tool "/Users/your_name/Desktop/My_Model_Package/fold_0/validation/summary.json"
```

或者：

```bash
grep -n '"CCC"' "/Users/your_name/Desktop/My_Model_Package/fold_0/validation/summary.json"
```


**八、如果你想重新跑一次验证集**

这一条只适合已经配好完整 nnU-Net 数据集环境的人。

如果你只是普通使用者，只看现成的 `validation/summary.json` 就够了，不需要自己重跑验证。

如果老师已经给你配置好了完整数据和环境，并且告诉你数据集编号、配置名、Trainer 名，那么可以用：

```bash
nnUNetv2_train 数据集ID 配置名 fold编号 -tr Trainer名称 -p nnUNetPlans --val --val_best -device cpu
```

一个示意例子：

```bash
nnUNetv2_train 701 2d 0 -tr MyTrainer_Attention -p nnUNetPlans --val --val_best -device cpu
```

说明：

- `--val`：只做验证，不重新训练
- `--val_best`：使用 `checkpoint_best.pth`

注意：

- 这个命令会重新写入 `validation` 文件夹内容
- 如果你不确定，就不要自己运行这一条


**九、如果命令报错，先检查这几件事**

1. 输入图像名字是否符合 nnU-Net 规则，例如 `case001_0000.nii.gz`
2. 输入图像后缀是否和训练时一致
3. `-m` 指向的是模型包上一级文件夹，而不是单独某个 `.pth`
4. 模型包里是否真的有 `fold_0/checkpoint_best.pth`
5. 电脑没有 GPU 时，是否写成了 `-device cpu`


**十、最推荐的实际使用方式**

对大多数同学，最推荐这样做：

1. 让老师提供完整模型文件夹，而不是单独 `.pth`
2. 把待预测图像放进一个单独文件夹
3. 在 PyCharm 的 Terminal 里运行预测命令
4. 预测结束后在输出文件夹查看结果
5. 如需查看验证集表现，打开 `fold_0/validation/summary.json` 搜索 `CCC`


**十一、老师给文件时的建议**

如果你是发模型给同学的人，建议提供以下内容：

- 完整模型文件夹，或导出的 zip 包
- 一个已经写好的预测命令示例
- 一张输入文件命名示例图
- 一句明确说明：推荐用 `checkpoint_best.pth`

不建议只发：

- `checkpoint_best.pth`

因为单独权重文件通常不足以完成 nnU-Net 推理。
