import argparse
from pathlib import Path
from pprint import pformat
from typing import List, Optional, Sequence

import torch

import nnunetv2
from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


"""
    我准备好的，可视化界面，nnUNet这个奇葩网络，很复杂很复杂很复杂。
    它有一个特点，不像我们以前看的那种简单模型，那种简单的模型，一般有一个model.py,里面写好了个各种代码啊，啥的，你在里面直接把人家的删除，再改成你的，就ok了。
    但是这个不一样，nnUNet的模型是怎么来的？我来简单梳理一下哈：
    nnUNet有一个很厉害很厉害的数据预处理部分，它会分析你的数据特点，比如脂肪数据，然后生成一个plans.json和dataset.json，根据这俩，做出一个它认为很好的网络架构。
    但是这有个问题，nnUNet想给你做的模型，你看不到，就很迷，所以我做了一个“可视化”模块，
    powershell输入python inspect_plans_model.py XXX XXX 啥的，就可以看到nnUNet想给你的数据集做的模型到底长啥样，这很方便，
    因为你不一定是每一次都需要自己做整个网络，那不现实，也很耗精力，我们每次只改动其中一个模块，我觉得再合适不过啦。
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description="读取 nnU-Net 的 plans.json，并构建对应模型用于查看结构。"
    )
    parser.add_argument(
        "plans",
        type=str,
        help="plans.json 或 nnUNetPlans.json 的路径。",
    )

    # 我推荐你最好把dataset.json的路径也提供一下，这样子好找。
    parser.add_argument(
        "-dj",
        "--dataset-json",
        type=str,
        default=None,
        help="可选的 dataset.json 路径。不填写时，脚本会尝试在 plans 文件附近自动查找。",
    )

    # 比如你是3d_fullres的配置，但想看看2d的结构，也可以通过这个参数指定。
    parser.add_argument(
        "-c",
        "--configuration",
        type=str,
        default=None,
        help="配置名，例如 2d / 3d_fullres。默认使用 plans 里的第一个 configuration。",
    )
    parser.add_argument(
        "-tr",
        "--trainer",
        type=str,
        default="MyTrainer_Attention",
        help="用来构建网络的 trainer 类名。默认：MyTrainer_Attention",
    )
    parser.add_argument(
        "--mode",
        choices=("trainer", "default", "both"),
        default="both",
        help="查看哪种模型视角：trainer=你的自定义 trainer，default=nnUNet 默认按 plans 搭建，both=两者都看。",
    )
    parser.add_argument(
        "--num-input-channels",
        type=int,
        default=None,
        help="当找不到 dataset.json 时，手动指定输入通道数。",
    )
    parser.add_argument(
        "--num-output-channels",
        type=int,
        default=None,
        help="当找不到 dataset.json 时，手动指定输出通道数。",
    )
    parser.add_argument(
        "--deep-supervision",
        choices=("auto", "on", "off"),
        default="auto",
        help="是否让 trainer 按 deep supervision 方式构建网络。默认：auto",
    )
    # 你最好找一个阳间的地方放着吧，不然每次运行都会在当前目录生成一大堆文件夹。
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="保存模型摘要和结构图的输出目录。默认：./model_inspect_<configuration>",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="额外尝试导出 hiddenlayer 的 PDF 结构图，以及 torch.fx 的图结构文本。",
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs="+",
        default=None,
        help="可选的完整假输入 shape，用于导图，例如：--input-shape 1 1 32 128 128",
    )
    return parser.parse_args()


def find_dataset_json(plans_path: Path, explicit_path: Optional[str]) -> Optional[Path]:
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"dataset.json not found: {path}")
        return path

    candidates = [
        plans_path.parent / "dataset.json",
        plans_path.parent.parent / "dataset.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_trainer_class(trainer_name: str):
    trainer_folder = join(nnunetv2.__path__[0], "training", "nnUNetTrainer")
    trainer_class = recursive_find_python_class(
        trainer_folder,
        trainer_name,
        current_module="nnunetv2.training.nnUNetTrainer",
    )
    if trainer_class is None:
        raise RuntimeError(
            f"Could not find trainer {trainer_name} under nnunetv2/training/nnUNetTrainer"
        )
    return trainer_class


def choose_configuration(plans_manager: PlansManager, requested: Optional[str]) -> str:
    if requested is not None:
        return requested
    return next(iter(plans_manager.plans["configurations"].keys()))


def choose_deep_supervision(mode: str, arch_init_kwargs: dict) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    return bool(arch_init_kwargs.get("deep_supervision", True))


def build_dummy_input_shape(
    explicit_shape: Optional[Sequence[int]],
    num_input_channels: int,
    patch_size: Sequence[int],
):
    if explicit_shape is not None:
        return tuple(explicit_shape)
    return (1, num_input_channels, *patch_size)


def export_hiddenlayer_graph(network: torch.nn.Module, dummy_input: torch.Tensor, output_file: Path):
    import hiddenlayer as hl

    graph = hl.build_graph(network, dummy_input, transforms=None)
    graph.save(str(output_file))


def export_fx_graph(network: torch.nn.Module, output_file: Path):
    traced = torch.fx.symbolic_trace(network)
    output_file.write_text(str(traced.graph), encoding="utf-8")


def ceil_div_tuple(values: Sequence[int], divisors: Sequence[int]) -> List[int]:
    return [int((v + d - 1) // d) for v, d in zip(values, divisors)]


def mul_tuple(values: Sequence[int], factors: Sequence[int]) -> List[int]:
    return [int(v * f) for v, f in zip(values, factors)]


def format_tensor_shape(channels: int, spatial: Sequence[int]) -> str:
    spatial_str = ", ".join(str(i) for i in spatial)
    return f"(B, {channels}, {spatial_str})"


def get_stage_role(module_name: str) -> str:
    if module_name == "Input":
        return "原始输入 patch"
    if "Encoder Stage 0" in module_name:
        return "浅层纹理特征"
    if "Encoder Stage 1" in module_name or "Encoder Stage 2" in module_name:
        return "逐步下采样，提取中层特征"
    if "Encoder Stage" in module_name:
        return "高语义特征，分辨率继续降低"
    if "Bottleneck" in module_name:
        return "语义最强的位置，最适合插 Attention/Transformer"
    if "Decoder Stage" in module_name:
        return "逐步上采样并融合 skip 特征"
    if "Seg Head" in module_name:
        return "输出最终分割 logits"
    if "Deep Supervision" in module_name:
        return "辅助监督分支"
    return ""


def build_plans_flow_entries(configuration_manager, num_input_channels: int, num_output_channels: int,
                             enable_deep_supervision: bool) -> List[dict]:
    arch_kwargs = configuration_manager.network_arch_init_kwargs
    features = arch_kwargs.get("features_per_stage", [])
    kernels = arch_kwargs.get("kernel_sizes", [])
    strides = arch_kwargs.get("strides", [])
    enc_depth = arch_kwargs.get("n_conv_per_stage", arch_kwargs.get("n_blocks_per_stage", []))
    dec_depth = arch_kwargs.get("n_conv_per_stage_decoder", [])
    patch_size = [int(i) for i in configuration_manager.patch_size]

    entries = []
    current_spatial = patch_size
    current_channels = num_input_channels

    entries.append({
        "module": "Input",
        "flow": format_tensor_shape(current_channels, current_spatial),
        "details": "输入 patch",
        "role": get_stage_role("Input"),
    })

    for idx, feat in enumerate(features):
        stride = strides[idx] if idx < len(strides) else [1] * len(current_spatial)
        kernel = kernels[idx] if idx < len(kernels) else "?"
        depth_here = enc_depth[idx] if idx < len(enc_depth) else "?"
        out_spatial = ceil_div_tuple(current_spatial, stride)
        stage_tag = "Bottleneck" if idx == len(features) - 1 else f"Encoder Stage {idx}"
        entries.append({
            "module": stage_tag,
            "flow": f"{format_tensor_shape(current_channels, current_spatial)} --> {format_tensor_shape(feat, out_spatial)}",
            "details": f"kernel={kernel}, stride={stride}, blocks/convs={depth_here}",
            "role": get_stage_role(stage_tag),
        })
        current_spatial = out_spatial
        current_channels = feat

    decoder_features = list(reversed(features[:-1]))
    decoder_strides = list(reversed(strides[1:])) if len(strides) > 1 else []
    for idx, feat in enumerate(decoder_features):
        stride = decoder_strides[idx] if idx < len(decoder_strides) else [1] * len(current_spatial)
        depth_here = dec_depth[idx] if idx < len(dec_depth) else "?"
        out_spatial = mul_tuple(current_spatial, stride)
        src_stage = len(decoder_features) - 1 - idx
        entries.append({
            "module": f"Decoder Stage {src_stage}",
            "flow": f"spatial {current_spatial} --> {out_spatial}",
            "details": f"输出通道={feat}, upsample={stride}, convs={depth_here}",
            "role": get_stage_role(f"Decoder Stage {src_stage}"),
        })
        current_spatial = out_spatial
        current_channels = feat

    entries.append({
        "module": "Seg Head",
        "flow": f"{format_tensor_shape(current_channels, current_spatial)} --> {format_tensor_shape(num_output_channels, current_spatial)}",
        "details": "分割输出头",
        "role": get_stage_role("Seg Head"),
    })
    entries.append({
        "module": "Deep Supervision",
        "flow": "enabled" if enable_deep_supervision else "disabled",
        "details": "多尺度监督开关",
        "role": get_stage_role("Deep Supervision"),
    })

    return entries


def build_plans_ascii_diagram(configuration_manager, num_input_channels: int, num_output_channels: int,
                              enable_deep_supervision: bool) -> str:
    entries = build_plans_flow_entries(
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
    )
    lines = ["模块流向图", "-" * 88]
    for entry in entries:
        lines.append(f"{entry['module']:<18} | {entry['flow']}")
        lines.append(f"{'':<18} | {entry['details']}")
    return "\n".join(lines)


def build_plans_markdown_table(configuration_manager, num_input_channels: int, num_output_channels: int,
                               enable_deep_supervision: bool) -> str:
    entries = build_plans_flow_entries(
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
    )
    lines = [
        "| 模块 | 维度变化 | 参数说明 | 教学备注 |",
        "| --- | --- | --- | --- |",
    ]
    for entry in entries:
        lines.append(
            f"| `{entry['module']}` | `{entry['flow']}` | {entry['details']} | {entry['role']} |"
        )
    return "\n".join(lines)


def build_basic_info_markdown_table(plans_path: Path, dataset_json_path: Optional[Path], trainer_name: str,
                                    configuration_name: str, configuration_manager, num_input_channels: int,
                                    num_output_channels: int, enable_deep_supervision: bool) -> str:
    lines = [
        "| 字段 | 值 |",
        "| --- | --- |",
        f"| plans 路径 | `{plans_path}` |",
        f"| dataset.json 路径 | `{dataset_json_path}` |" if dataset_json_path is not None else "| dataset.json 路径 | `未提供` |",
        f"| trainer | `{trainer_name}` |",
        f"| configuration | `{configuration_name}` |",
        f"| patch_size | `{configuration_manager.patch_size}` |",
        f"| spacing | `{configuration_manager.spacing}` |",
        f"| network_arch_class_name | `{configuration_manager.network_arch_class_name}` |",
        f"| num_input_channels | `{num_input_channels}` |",
        f"| num_output_channels | `{num_output_channels}` |",
        f"| enable_deep_supervision | `{enable_deep_supervision}` |",
    ]
    return "\n".join(lines)


def build_teaching_flow_text(configuration_manager, num_input_channels: int, num_output_channels: int,
                             enable_deep_supervision: bool) -> str:
    entries = build_plans_flow_entries(
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
    )
    lines = [
        "一眼看懂版",
        "=" * 88,
    ]
    for idx, entry in enumerate(entries):
        if idx == 0:
            lines.append(f"[{entry['module']}] {entry['flow']}")
        else:
            lines.append(f"   --> [{entry['module']}] {entry['flow']}")
        if entry["role"]:
            lines.append(f"       作用: {entry['role']}")
    return "\n".join(lines)


def build_teaching_flow_markdown(configuration_manager, num_input_channels: int, num_output_channels: int,
                                 enable_deep_supervision: bool) -> str:
    entries = build_plans_flow_entries(
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
    )
    lines = []
    for idx, entry in enumerate(entries):
        prefix = "" if idx == 0 else "&nbsp;&nbsp;&nbsp;--> "
        line = f"{prefix}**{entry['module']}**: `{entry['flow']}`"
        if entry["role"]:
            line += f"  \n{prefix}说明: {entry['role']}"
        lines.append(line)
    return "\n\n".join(lines)


def sanitize_mermaid_label(text: str) -> str:
    return text.replace('"', "'")


def build_mermaid_diagram(configuration_manager, num_input_channels: int, num_output_channels: int,
                          enable_deep_supervision: bool) -> str:
    entries = build_plans_flow_entries(
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
    )
    lines = ["flowchart TD"]
    for idx, entry in enumerate(entries):
        node_id = f"N{idx}"
        role = f"<br/>{entry['role']}" if entry["role"] else ""
        label = sanitize_mermaid_label(f"{entry['module']}<br/>{entry['flow']}{role}")
        lines.append(f'    {node_id}["{label}"]')
        if idx > 0:
            lines.append(f"    N{idx - 1} --> {node_id}")
    return "\n".join(lines)


def describe_module(module: torch.nn.Module) -> str:
    if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
        return (
            f"{module.__class__.__name__}"
            f"({module.in_channels}->{module.out_channels}, "
            f"k={tuple(module.kernel_size)}, s={tuple(module.stride)})"
        )
    if isinstance(module, torch.nn.Linear):
        return f"Linear({module.in_features}->{module.out_features})"
    if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
                           torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d)):
        return f"{module.__class__.__name__}(num_features={module.num_features})"
    if isinstance(module, torch.nn.Sequential):
        return f"Sequential(len={len(module)})"
    return module.__class__.__name__


def build_module_tree_lines(module: torch.nn.Module, prefix: str = "", depth: int = 0, max_depth: int = 3) -> List[str]:
    if depth == 0:
        lines = [describe_module(module)]
    else:
        lines = []

    if depth >= max_depth:
        return lines

    children = list(module.named_children())
    for name, child in children:
        lines.append(f"{prefix}-> {name}: {describe_module(child)}")
        lines.extend(build_module_tree_lines(child, prefix + "   ", depth + 1, max_depth))
    return lines


def build_network_section_markdown(title: str, label: str, network: torch.nn.Module) -> str:
    tree_text = "\n".join(build_module_tree_lines(network))
    return "\n".join([
        f"## {title}",
        "",
        f"- 视角说明: `{label}`",
        "",
        "### 模块树",
        "```text",
        tree_text,
        "```",
        "",
        "### 完整模型打印",
        "```python",
        str(network),
        "```",
        "",
    ])


def make_plain_summary(
    plans_path: Path,
    dataset_json_path: Optional[Path],
    trainer_name: str,
    configuration_name: str,
    configuration_manager,
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool,
    network_views: List[dict],
) -> str:
    arch_kwargs_text = pformat(
        configuration_manager.network_arch_init_kwargs,
        sort_dicts=False,
        width=100,
    )
    req_import_text = pformat(
        configuration_manager.network_arch_init_kwargs_req_import,
        sort_dicts=False,
        width=100,
    )

    lines = [
        "=" * 88,
        "模型检查报告",
        "=" * 88,
        "",
        "[基础信息]",
        f"- plans 路径: {plans_path}",
        f"- dataset.json 路径: {dataset_json_path if dataset_json_path is not None else '未提供'}",
        f"- trainer: {trainer_name}",
        f"- configuration: {configuration_name}",
        f"- patch_size: {configuration_manager.patch_size}",
        f"- spacing: {configuration_manager.spacing}",
        f"- network_arch_class_name: {configuration_manager.network_arch_class_name}",
        f"- num_input_channels: {num_input_channels}",
        f"- num_output_channels: {num_output_channels}",
        f"- enable_deep_supervision: {enable_deep_supervision}",
        "",
        "[plans 级骨架示意]",
        build_plans_ascii_diagram(
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        ),
        "",
        "[教学图：一眼看懂版]",
        build_teaching_flow_text(
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        ),
        "",
        "[network_arch_init_kwargs]",
        arch_kwargs_text,
        "",
        "[network_arch_init_kwargs_req_import]",
        req_import_text,
        "",
        "[说明]",
        "- 上面的维度变化是根据 plans 里的 stride / patch_size 推算出来的整体流向图。",
        "- Decoder 部分展示的是空间尺寸如何变化，方便你定位该在哪一层插模块。",
        "",
    ]

    for view in network_views:
        lines.extend([
            f"[{view['title']}]",
            f"- 视角说明: {view['label']}",
            "",
            "[模块树]",
            "\n".join(build_module_tree_lines(view["network"])),
            "",
            "[完整模型结构]",
            str(view["network"]),
            "",
        ])
    return "\n".join(lines)


def make_markdown_summary(
    plans_path: Path,
    dataset_json_path: Optional[Path],
    trainer_name: str,
    configuration_name: str,
    configuration_manager,
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool,
    network_views: List[dict],
) -> str:
    arch_kwargs_text = pformat(
        configuration_manager.network_arch_init_kwargs,
        sort_dicts=False,
        width=100,
    )
    req_import_text = pformat(
        configuration_manager.network_arch_init_kwargs_req_import,
        sort_dicts=False,
        width=100,
    )

    lines = [
        "# 模型检查报告",
        "",
        "## 基础信息",
        build_basic_info_markdown_table(
            plans_path,
            dataset_json_path,
            trainer_name,
            configuration_name,
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        ),
        "",
        "## plans 级骨架示意",
        "> 这一部分是按 plans 推出来的结构流向图，重点看每个模块前后 shape 怎么变。",
        "",
        "```text",
        build_plans_ascii_diagram(
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        ),
        "```",
        "",
        "## 教学图：一眼看懂版",
        "> 这一部分更像讲义，适合先快速理解网络大框架，再去看底下的完整模型树。",
        "",
        build_teaching_flow_markdown(
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        ),
        "",
        "## 教学图：Mermaid 版",
        "> 如果你的 Markdown 预览支持 Mermaid，这一段会渲染成真正的流程图。",
        "",
        "```mermaid",
        build_mermaid_diagram(
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        ),
        "```",
        "",
        "## plans 级维度变化表",
        build_plans_markdown_table(
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        ),
        "",
        "## network_arch_init_kwargs",
        "```python",
        arch_kwargs_text,
        "```",
        "",
        "## network_arch_init_kwargs_req_import",
        "```python",
        req_import_text,
        "```",
        "",
    ]
    for view in network_views:
        lines.append(build_network_section_markdown(view["title"], view["label"], view["network"]))
    return "\n".join(lines)


def build_network_with_trainer(trainer_name: str, configuration_manager, num_input_channels: int,
                               num_output_channels: int, enable_deep_supervision: bool) -> torch.nn.Module:
    trainer_class = resolve_trainer_class(trainer_name)
    network = trainer_class.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
    )
    return network.cpu().eval()


def build_default_network(configuration_manager, num_input_channels: int, num_output_channels: int,
                          enable_deep_supervision: bool) -> torch.nn.Module:
    network = nnUNetTrainer.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
    )
    return network.cpu().eval()


def main():
    args = parse_args()

    plans_path = Path(args.plans).expanduser().resolve()
    if not plans_path.is_file():
        raise FileNotFoundError(f"plans file not found: {plans_path}")

    dataset_json_path = find_dataset_json(plans_path, args.dataset_json)

    plans = load_json(str(plans_path))
    plans_manager = PlansManager(plans)
    configuration_name = choose_configuration(plans_manager, args.configuration)
    configuration_manager = plans_manager.get_configuration(configuration_name)

    dataset_json = load_json(str(dataset_json_path)) if dataset_json_path is not None else None

    if dataset_json is not None:
        num_input_channels = determine_num_input_channels(
            plans_manager,
            configuration_manager,
            dataset_json,
        )
        num_output_channels = plans_manager.get_label_manager(dataset_json).num_segmentation_heads
    else:
        if args.num_input_channels is None or args.num_output_channels is None:
            raise RuntimeError(
                "dataset.json was not found, so you must provide both "
                "--num-input-channels and --num-output-channels."
            )
        num_input_channels = args.num_input_channels
        num_output_channels = args.num_output_channels

    enable_deep_supervision = choose_deep_supervision(
        args.deep_supervision,
        configuration_manager.network_arch_init_kwargs,
    )

    network_views: List[dict] = []
    if args.mode in ("trainer", "both"):
        network_views.append({
            "title": "自定义 Trainer 视角",
            "label": f"{args.trainer} 实际返回的模型",
            "name": "trainer",
            "network": build_network_with_trainer(
                args.trainer,
                configuration_manager,
                num_input_channels,
                num_output_channels,
                enable_deep_supervision,
            ),
        })
    if args.mode in ("default", "both"):
        network_views.append({
            "title": "nnUNet 默认视角",
            "label": "严格按照 plans.json + dataset.json 构建的默认网络",
            "name": "default",
            "network": build_default_network(
                configuration_manager,
                num_input_channels,
                num_output_channels,
                enable_deep_supervision,
            ),
        })

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir is not None else (
        Path.cwd() / f"model_inspect_{configuration_name}"
    )
    maybe_mkdir_p(str(output_dir))

    summary_text = make_plain_summary(
        plans_path,
        dataset_json_path,
        args.trainer,
        configuration_name,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
        network_views,
    )
    summary_md = make_markdown_summary(
        plans_path,
        dataset_json_path,
        args.trainer,
        configuration_name,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
        network_views,
    )
    summary_file = output_dir / "model_summary.txt"
    summary_md_file = output_dir / "model_summary.md"
    summary_file.write_text(summary_text, encoding="utf-8")
    summary_md_file.write_text(summary_md, encoding="utf-8")

    print(summary_text)
    print(f"Saved model summary to: {summary_file}")
    print(f"Saved markdown summary to: {summary_md_file}")

    if args.graph:
        dummy_input_shape = build_dummy_input_shape(
            args.input_shape,
            num_input_channels,
            configuration_manager.patch_size,
        )
        dummy_input = torch.randn(dummy_input_shape, dtype=torch.float32)
        for view in network_views:
            fx_graph_file = output_dir / f"{view['name']}_fx_graph.txt"
            try:
                export_fx_graph(view["network"], fx_graph_file)
                print(f"Saved torch.fx graph to: {fx_graph_file}")
            except Exception as e:
                print(f"{view['name']} 的 torch.fx 图导出跳过: {e}")

            pdf_graph_file = output_dir / f"{view['name']}_network_architecture.pdf"
            try:
                with torch.no_grad():
                    export_hiddenlayer_graph(view["network"], dummy_input, pdf_graph_file)
                print(f"Saved hiddenlayer PDF graph to: {pdf_graph_file}")
            except ModuleNotFoundError:
                print("hiddenlayer 未安装，因此 PDF 结构图导出被跳过。")
                break
            except Exception as e:
                print(f"{view['name']} 的 hiddenlayer 结构图导出跳过: {e}")


if __name__ == "__main__":
    main()
