from nnunetv2.configuration import default_num_processes
from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess


def extract_fingerprint_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+', type=int,
                        help="[必填] 数据集 ID 列表，例如：2 4 5。会对这些数据集执行 fingerprint 提取、实验规划和预处理，当然也可以只写一个数据集。")
    parser.add_argument('-fpe', type=str, required=False, default='DatasetFingerprintExtractor',
                        help='[可选] 要使用的 Dataset Fingerprint Extractor 类名。默认：DatasetFingerprintExtractor。')
    parser.add_argument('-np', type=int, default=default_num_processes, required=False,
                        help=f'[可选] fingerprint 提取使用的进程数。默认：{default_num_processes}')
    parser.add_argument("--verify_dataset_integrity", required=False, default=False, action="store_true",
                        help="[推荐] 设置此参数以检查数据集完整性。每个数据集建议至少做一次！")
    parser.add_argument("--clean", required=False, default=False, action="store_true",
                        help='[可选] 覆盖已有的 fingerprint。如果不设置且 fingerprint 已存在，则不会重新运行 fingerprint extractor。')
    parser.add_argument('--verbose', required=False, action='store_true',
                        help='设置后会打印大量信息，适合调试；同时会关闭进度条。推荐在集群环境中使用。')
    args, unrecognized_args = parser.parse_known_args()
    extract_fingerprints(args.d, args.fpe, args.np, args.verify_dataset_integrity, args.clean, args.verbose)


def plan_experiment_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+', type=int,
                        help="[必填] 数据集 ID 列表，例如：2 4 5。会对这些数据集执行 fingerprint 提取、实验规划和预处理，当然也可以只写一个数据集。")
    parser.add_argument('-pl', type=str, default='ExperimentPlanner', required=False,
                        help='[可选] 要使用的 Experiment Planner 类名。默认：ExperimentPlanner。注意：现在不再区分 2d planner 和 3d planner，而是统一方案。')
    parser.add_argument('-gpu_memory_target', default=None, type=float, required=False,
                        help='[可选][危险区域] 自定义 GPU 显存目标（单位 GB）。默认：None（即使用 Planner 类默认值）。修改它会影响 patch size 和 batch size，并且肯定会影响模型性能！只有在你非常清楚自己在做什么时才使用，并且一定要先跑默认 nnU-Net 作为 baseline。')
    parser.add_argument('-preprocessor_name', default='DefaultPreprocessor', type=str, required=False,
                        help='[可选][危险区域] 自定义 preprocessor 类名。该类必须位于 nnunetv2.preprocessing 中。默认：DefaultPreprocessor。修改它可能会影响模型性能！只有在你非常清楚自己在做什么时才使用，并且一定要先跑默认 nnU-Net 作为 baseline。')
    parser.add_argument('-overwrite_target_spacing', default=None, nargs='+', required=False,
                        help='[可选][危险区域] 为 3d_fullres 和 3d_cascade_fullres 配置自定义 target spacing。默认：None（不修改）。改变它会影响图像大小，也可能影响 patch size 和 batch size，并且一定会影响模型性能！只有在你非常清楚自己在做什么时才使用，并且一定要先跑默认 nnU-Net 作为 baseline。其他 configuration 的 target spacing 目前不支持修改。新的 target spacing 必须是 3 个数字。')
    parser.add_argument('-overwrite_plans_name', default=None, required=False,
                        help='[可选][危险区域] 如果你用了 -gpu_memory_target、-preprocessor_name 或 -overwrite_target_spacing，最佳实践是同时用 -overwrite_plans_name 生成一个不同名字的 plans 文件，避免覆盖 nnU-Net 默认 plans。之后在运行其他 nnU-Net 命令（训练、推理等）时，需要通过 -p 指定你的自定义 plans。')
    args, unrecognized_args = parser.parse_known_args()
    plan_experiments(args.d, args.pl, args.gpu_memory_target, args.preprocessor_name, args.overwrite_target_spacing,
                     args.overwrite_plans_name)


def preprocess_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+', type=int,
                        help="[必填] 数据集 ID 列表，例如：2 4 5。会对这些数据集执行 fingerprint 提取、实验规划和预处理，当然也可以只写一个数据集。")
    parser.add_argument('-plans_name', default='nnUNetPlans', required=False,
                        help='[可选] 指定你自己生成的 plans 文件名称。')
    parser.add_argument('-c', required=False, default=['2d', '3d_fullres', '3d_lowres'], nargs='+',
                        help='[可选] 指定需要运行预处理的 configuration。默认：2d 3d_fullres 3d_lowres。3d_cascade_fullres 不需要单独指定，因为它复用 3d_fullres 的数据。某些数据集里不存在的 configuration 会被自动跳过。')
    parser.add_argument('-np', type=int, nargs='+', default=None, required=False,
                        help="[可选] 指定使用多少个进程。如果这里只给一个数字，那么 -c 中所有 configuration 都会使用这个进程数；如果给的是一串数字，那么它的长度必须和 configuration 数量一致，程序会按 zip(configs, num_processes) 逐一对应。进程多通常更快，但上限受 CPU 线程数和内存限制影响。警告：图像重采样非常吃内存，请务必监控 RAM，如果内存占用过高就降低 -np。默认：2d 用 8，3d_fullres 用 4，3d_lowres 用 8，其余用 4。")
    parser.add_argument('--verbose', required=False, action='store_true',
                        help='设置后会打印大量信息，适合调试；同时会关闭进度条。推荐在集群环境中使用。')
    args, unrecognized_args = parser.parse_known_args()
    if args.np is None:
        default_np = {"2d": 8, "3d_fullres": 4, "3d_lowres": 8}
        np = [default_np[c] if c in default_np.keys() else 4 for c in args.c]
    else:
        np = args.np
    preprocess(args.d, args.plans_name, configurations=args.c, num_processes=np, verbose=args.verbose)


# 在这里！！！！
def plan_and_preprocess_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+', type=int,
                        help="[必填] 数据集 ID 列表，例如：2 4 5。会对这些数据集执行 fingerprint 提取、实验规划和预处理，当然也可以只写一个数据集。")
    parser.add_argument('-fpe', type=str, required=False, default='DatasetFingerprintExtractor',
                        help='[可选] 要使用的 Dataset Fingerprint Extractor 类名。默认：DatasetFingerprintExtractor。')
    parser.add_argument('-npfp', type=int, default=8, required=False,
                        help='[可选] fingerprint 提取使用的进程数。默认：8。')
    parser.add_argument("--verify_dataset_integrity", required=False, default=False, action="store_true",
                        help="[推荐] 设置此参数以检查数据集完整性。每个数据集建议至少做一次！")
    parser.add_argument('--no_pp', default=False, action='store_true', required=False,
                        help='[可选] 只运行 fingerprint 提取和实验规划（不做预处理）。适合调试。')
    parser.add_argument("--clean", required=False, default=False, action="store_true",
                        help='[可选] 覆盖已有 fingerprint。如果不设置且 fingerprint 已存在，则不会重新运行 fingerprint extractor。如果你改了 Dataset Fingerprint Extractor 或修改了数据集，这个参数就是必须的！')
    parser.add_argument('-pl', type=str, default='ExperimentPlanner', required=False,
                        help='[可选] 要使用的 Experiment Planner 类名。默认：ExperimentPlanner。注意：现在不再区分 2d planner 和 3d planner，而是统一方案。')
    parser.add_argument('-gpu_memory_target', default=None, type=float, required=False,
                        help='[可选][危险区域] 自定义 GPU 显存目标（单位 GB）。默认：None（即使用 Planner 类默认值）。修改它会影响 patch size 和 batch size，并且肯定会影响模型性能！只有在你非常清楚自己在做什么时才使用，并且一定要先跑默认 nnU-Net 作为 baseline。')
    parser.add_argument('-preprocessor_name', default='DefaultPreprocessor', type=str, required=False,
                        help='[可选][危险区域] 自定义 preprocessor 类名。该类必须位于 nnunetv2.preprocessing 中。默认：DefaultPreprocessor。修改它可能会影响模型性能！只有在你非常清楚自己在做什么时才使用，并且一定要先跑默认 nnU-Net 作为 baseline。')
    parser.add_argument('-overwrite_target_spacing', default=None, nargs='+', required=False,
                        help='[可选][危险区域] 为 3d_fullres 和 3d_cascade_fullres 配置自定义 target spacing。默认：None（不修改）。改变它会影响图像大小，也可能影响 patch size 和 batch size，并且一定会影响模型性能！只有在你非常清楚自己在做什么时才使用，并且一定要先跑默认 nnU-Net 作为 baseline。其他 configuration 的 target spacing 目前不支持修改。新的 target spacing 必须是 3 个数字。')
    parser.add_argument('-overwrite_plans_name', default=None, required=False,
                        help='[可选] 使用自定义 plans 标识符。如果你用了 -gpu_memory_target、-preprocessor_name 或 -overwrite_target_spacing，最佳实践是同时用 -overwrite_plans_name 生成一个不同名字的 plans 文件，避免覆盖 nnU-Net 默认 plans。之后在运行其他 nnU-Net 命令（训练、推理等）时，需要通过 -p 指定你的自定义 plans。')
    parser.add_argument('-c', required=False, default=['2d', '3d_fullres', '3d_lowres'], nargs='+',
                        help='[可选] 指定需要运行预处理的 configuration。默认：2d 3d_fullres 3d_lowres。3d_cascade_fullres 不需要单独指定，因为它复用 3d_fullres 的数据。某些数据集里不存在的 configuration 会被自动跳过。')
    parser.add_argument('-np', type=int, nargs='+', default=None, required=False,
                        help="[可选] 指定使用多少个进程。如果这里只给一个数字，那么 -c 中所有 configuration 都会使用这个进程数；如果给的是一串数字，那么它的长度必须和 configuration 数量一致，程序会按 zip(configs, num_processes) 逐一对应。进程多通常更快，但上限受 CPU 线程数和内存限制影响。警告：图像重采样非常吃内存，请务必监控 RAM，如果内存占用过高就降低 -np。默认：2d 用 8，3d_fullres 用 4，3d_lowres 用 8，其余用 4。")
    parser.add_argument('--verbose', required=False, action='store_true',
                        help='设置后会打印大量信息，适合调试；同时会关闭进度条。推荐在集群环境中使用。')
    args = parser.parse_args()

    # fingerprint extraction
    print("Fingerprint extraction...")
    extract_fingerprints(args.d, args.fpe, args.npfp, args.verify_dataset_integrity, args.clean, args.verbose)

    # experiment planning
    print('Experiment planning...')
    plans_identifier = plan_experiments(args.d, args.pl, args.gpu_memory_target, args.preprocessor_name,
                                        args.overwrite_target_spacing, args.overwrite_plans_name)

    # manage default np
    if args.np is None:
        default_np = {"2d": 8, "3d_fullres": 4, "3d_lowres": 8}
        np = [default_np[c] if c in default_np.keys() else 4 for c in args.c]
    else:
        np = args.np
    # preprocessing
    if not args.no_pp:
        print('Preprocessing...')
        preprocess(args.d, plans_identifier, args.c, np, args.verbose)


if __name__ == '__main__':
    plan_and_preprocess_entry()
