import argparse
from pathlib import Path
from typing import Optional, Sequence

import torch

import nnunetv2
from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
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
    trainer_class = resolve_trainer_class(args.trainer)

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

    network = trainer_class.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
    )
    network = network.cpu().eval()

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir is not None else (
        Path.cwd() / f"model_inspect_{configuration_name}"
    )
    maybe_mkdir_p(str(output_dir))

    summary_lines = [
        f"plans: {plans_path}",
        f"dataset_json: {dataset_json_path if dataset_json_path is not None else 'not provided'}",
        f"trainer: {args.trainer}",
        f"configuration: {configuration_name}",
        f"patch_size: {configuration_manager.patch_size}",
        f"spacing: {configuration_manager.spacing}",
        f"network_arch_class_name: {configuration_manager.network_arch_class_name}",
        f"num_input_channels: {num_input_channels}",
        f"num_output_channels: {num_output_channels}",
        f"enable_deep_supervision: {enable_deep_supervision}",
        "",
        "network_arch_init_kwargs:",
        str(configuration_manager.network_arch_init_kwargs),
        "",
        "network_arch_init_kwargs_req_import:",
        str(configuration_manager.network_arch_init_kwargs_req_import),
        "",
        "model:",
        str(network),
        "",
    ]

    summary_text = "\n".join(summary_lines)
    summary_file = output_dir / "model_summary.txt"
    summary_file.write_text(summary_text, encoding="utf-8")

    print(summary_text)
    print(f"Saved model summary to: {summary_file}")

    if args.graph:
        dummy_input_shape = build_dummy_input_shape(
            args.input_shape,
            num_input_channels,
            configuration_manager.patch_size,
        )
        dummy_input = torch.randn(dummy_input_shape, dtype=torch.float32)

        fx_graph_file = output_dir / "fx_graph.txt"
        try:
            export_fx_graph(network, fx_graph_file)
            print(f"Saved torch.fx graph to: {fx_graph_file}")
        except Exception as e:
            print(f"torch.fx graph export skipped: {e}")

        pdf_graph_file = output_dir / "network_architecture.pdf"
        try:
            with torch.no_grad():
                export_hiddenlayer_graph(network, dummy_input, pdf_graph_file)
            print(f"Saved hiddenlayer PDF graph to: {pdf_graph_file}")
        except ModuleNotFoundError:
            print("hiddenlayer is not installed, so PDF graph export was skipped.")
        except Exception as e:
            print(f"hiddenlayer graph export skipped: {e}")


if __name__ == "__main__":
    main()
