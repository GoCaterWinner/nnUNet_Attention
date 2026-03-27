from nnunetv2.training.nnUNetTrainer.trainer_attention import MyTrainer_Attention
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
import json
import os

# 这个是用来帮你测试你的 MyTrainer_Attention 的 build_network_architecture 里到底有没有问题的，就是能不能走通到那里。
def main():
    # 1) 找到你训练时用的 plans 和 dataset.json
    # 这些通常在 nnUNet_results 里的某个实验文件夹下
    # 你先把路径填成你电脑上真实存在的
    plans_path = r"E:\nnU-Net-Data\nnUNet_results\Dataset001_Lung\nnUNetTrainer__nnUNetPlans__2d\plans.json"
    dataset_json_path = r"E:\nnU-Net-Data\nnUNet_results\Dataset001_Lung\nnUNetTrainer__nnUNetPlans__2d\dataset.json"

    with open(plans_path, "r", encoding="utf-8") as f:
        plans = json.load(f)
    with open(dataset_json_path, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)

    # 2) 随便挑一个 configuration / fold
    configuration = list(plans["configurations"].keys())[0]
    fold = 0

    trainer = MyTrainer_Attention(plans, configuration, fold, dataset_json)

    # 3) 走到 build_network_architecture（print 就会出）
    trainer.initialize()

if __name__ == "__main__":
    main()