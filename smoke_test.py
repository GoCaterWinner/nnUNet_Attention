import torch
from nnunetv2.training.nnUNetTrainer.trainer_attention import MyTrainer_Attention

def main():
    # 直接调用静态方法构造网络（模拟 nnU-Net 调用方式）
    net = MyTrainer_Attention.build_network_architecture(
        architecture_class_name="whatever",
        arch_init_kwargs={},  # 最小先空
        arch_init_kwargs_req_import=[],
        num_input_channels=1,
        num_output_channels=3,
        enable_deep_supervision=True,
    )

    x = torch.randn(2, 1, 128, 128)  # (B,C,H,W)
    y = net(x)

    print("forward ok, type:", type(y))
    if isinstance(y, list):
        print("deep supervision outputs:", [t.shape for t in y])
    else:
        print("output shape:", y.shape)

if __name__ == "__main__":
    main()