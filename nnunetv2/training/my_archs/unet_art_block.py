import torch
import torch.nn as nn


class _DecoderShim:
    """
    给自定义网络提供一个和 nnU-Net 默认网络一致的 `decoder.deep_supervision` 入口。
    这样 Trainer 在切换 train/val 模式时，不会因为找不到该属性而报错。
    """
    def __init__(self, deep_supervision: bool = True):
        self.deep_supervision = deep_supervision


class ARTRefineBlock(nn.Module):
    """
    一个轻量版 ART 风格注意力细化块（支持 2D/3D）。

    设计目标：
    1. 尽量不改动 nnU-Net 主干网络，只在输出端做“缝合”；
    2. 参数量可控，避免显著拖慢训练；
    3. 既能处理最终输出，也能处理 deep supervision 的多尺度输出。
    """
    def __init__(self, channels: int, is_3d: bool):
        super().__init__()
        conv = nn.Conv3d if is_3d else nn.Conv2d
        norm = nn.InstanceNorm3d if is_3d else nn.InstanceNorm2d
        hidden = max(8, channels * 2)

        # 先做通道投影，再计算注意力门控，最后残差回加到原 logits
        self.pre = conv(channels, hidden, kernel_size=1, bias=True)
        self.mix = conv(hidden, hidden, kernel_size=3, padding=1, bias=False)
        self.norm = norm(hidden, affine=True)
        self.act = nn.GELU()
        self.gate = conv(hidden, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.pre(x)
        attn = self.mix(attn)
        attn = self.norm(attn)
        attn = self.act(attn)
        attn = self.gate(attn)
        attn = self.sigmoid(attn)
        # 残差式细化：保留原预测，同时用注意力门控突出关键区域
        return x + x * attn


class UNetARTBlock(nn.Module):
    """
    把 ART 细化块“包裹”到 nnU-Net 主干网络输出端的适配器。

    接口约定：
    - 输入：原始图像张量 x；
    - 输出：
      - 若主干开启 deep supervision：返回 list[Tensor]
      - 否则：返回单个 Tensor
    - Trainer 会通过 `decoder.deep_supervision` 来切换网络输出形式，
      所以这里也提供同名入口并同步到底层主干网络。
    """
    def __init__(self, backbone: nn.Module, num_classes: int, deep_supervision: bool):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        # 与 nnU-Net 的约定保持一致：Trainer 会访问 self.network.decoder.deep_supervision
        self.decoder = _DecoderShim(deep_supervision=deep_supervision)

        # 自动探测主干是 2D 还是 3D，避免写死 Conv2d 导致 3d_fullres 崩溃
        is_3d = any(isinstance(m, nn.Conv3d) for m in self.backbone.modules())
        self.refine = ARTRefineBlock(channels=num_classes, is_3d=is_3d)
        self.set_deep_supervision(deep_supervision)

    def set_deep_supervision(self, enabled: bool) -> None:
        """
        统一管理 deep supervision 开关：
        - 更新包装层的 shim；
        - 同步到底层主干（如果主干也有 decoder.deep_supervision）。
        """
        self.decoder.deep_supervision = enabled
        if hasattr(self.backbone, "decoder") and hasattr(self.backbone.decoder, "deep_supervision"):
            self.backbone.decoder.deep_supervision = enabled

    def _refine_output(self, out):
        if isinstance(out, (list, tuple)):
            # deep supervision 场景：每个尺度的 logits 都做一次 ART 细化
            return [self.refine(o) for o in out]
        return self.refine(out)

    def forward(self, x: torch.Tensor):
        out = self.backbone(x)
        return self._refine_output(out)

