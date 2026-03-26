import torch
from nnunetv2.training.nnUNetTrainer.trainer_attention import MyTrainer_Attention


# 最小可跑通单元，意思就是你把网络做好了，这玩意可以帮你检测，你的网络有没有语法错误，有没有逻辑上的错误？
# 能否顺利跑满一轮，也就是用来做小测试的.
def main():
    # 当前 YourNet 用的是 Conv3d，所以这里要给 5D 输入:
    # (B, C, D, H, W)
    net = MyTrainer_Attention.build_network_architecture(
        architecture_class_name="unused_by_YourNet",
        arch_init_kwargs={},  # YourNet 当前会吃掉多余 kwargs，这里最小先空
        arch_init_kwargs_req_import=[],
        num_input_channels=1,
        num_output_channels=3,
        enable_deep_supervision=False,
    )

    net.eval()
    # 这个就是一个最小的数据，帮你做测试的。
    x = torch.randn(2, 1, 16, 64, 64)  # (B,C,D,H,W)
    with torch.no_grad():
        y = net(x)

    print("forward ok, type:", type(y))
    if isinstance(y, list):
        print("deep supervision outputs:", [tuple(t.shape) for t in y])
    else:
        print("input shape:", tuple(x.shape))
        print("output shape:", tuple(y.shape))


if __name__ == "__main__":
    main()
