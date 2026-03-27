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


def build_plans_ascii_diagram(configuration_manager, num_input_channels: int, num_output_channels: int,
                              enable_deep_supervision: bool) -> str:
    arch_kwargs = configuration_manager.network_arch_init_kwargs
    features = arch_kwargs.get("features_per_stage", [])
    kernels = arch_kwargs.get("kernel_sizes", [])
    strides = arch_kwargs.get("strides", [])
    enc_depth = arch_kwargs.get("n_conv_per_stage", arch_kwargs.get("n_blocks_per_stage", []))
    dec_depth = arch_kwargs.get("n_conv_per_stage_decoder", [])
    patch_size = configuration_manager.patch_size

    lines = [
        f"输入(B, {num_input_channels}, {', '.join(str(i) for i in patch_size)})",
    ]

    for idx, feat in enumerate(features):
        stage_tag = "Bottleneck" if idx == len(features) - 1 else f"Encoder Stage {idx}"
        kernel = kernels[idx] if idx < len(kernels) else "?"
        stride = strides[idx] if idx < len(strides) else "?"
        depth_here = enc_depth[idx] if idx < len(enc_depth) else "?"
        lines.append(
            f"  -> {stage_tag} [通道={feat}, kernel={kernel}, stride={stride}, blocks/convs={depth_here}]"
        )

    decoder_features = list(reversed(features[:-1]))
    for idx, feat in enumerate(decoder_features):
        src_stage = len(decoder_features) - 1 - idx
        depth_here = dec_depth[idx] if idx < len(dec_depth) else "?"
        lines.append(
            f"  -> Decoder Stage {src_stage} [通道={feat}, convs={depth_here}]"
        )

    lines.append(f"  -> Seg Head [输出通道={num_output_channels}]")
    if enable_deep_supervision:
        lines.append("  -> Deep Supervision Heads [开启]")
    else:
        lines.append("  -> Deep Supervision Heads [关闭]")
    return "\n".join(lines)


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
        "[network_arch_init_kwargs]",
        arch_kwargs_text,
        "",
        "[network_arch_init_kwargs_req_import]",
        req_import_text,
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
        f"- plans 路径: `{plans_path}`",
        f"- dataset.json 路径: `{dataset_json_path}`" if dataset_json_path is not None else "- dataset.json 路径: `未提供`",
        f"- trainer: `{trainer_name}`",
        f"- configuration: `{configuration_name}`",
        f"- patch_size: `{configuration_manager.patch_size}`",
        f"- spacing: `{configuration_manager.spacing}`",
        f"- network_arch_class_name: `{configuration_manager.network_arch_class_name}`",
        f"- num_input_channels: `{num_input_channels}`",
        f"- num_output_channels: `{num_output_channels}`",
        f"- enable_deep_supervision: `{enable_deep_supervision}`",
        "",
        "## plans 级骨架示意",
        "```text",
        build_plans_ascii_diagram(
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        ),
        "```",
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
