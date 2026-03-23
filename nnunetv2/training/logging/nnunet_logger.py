import matplotlib
from batchgenerators.utilities.file_and_folder_operations import join

matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt


class nnUNetLogger(object):
    """
    This class is really trivial. Don't expect cool functionality here. This is my makeshift solution to problems
    arising from out-of-sync epoch numbers and numbers of logged loss values. It also simplifies the trainer class a
    little

    YOU MUST LOG EXACTLY ONE VALUE PER EPOCH FOR EACH OF THE LOGGING ITEMS! DONT FUCK IT UP

    新增：
    - val_ccc: 每个 epoch 验证集上的 pseudo CCC（体积一致性相关系数）
    - val_hd95: 每个 epoch 验证集上的 pseudo HD95
    """
    def __init__(self, verbose: bool = False):
        self.my_fantastic_logging = {
            'mean_fg_dice': list(),
            'ema_fg_dice': list(),
            'dice_per_class_or_region': list(),
            'train_losses': list(),
            'val_losses': list(),
            'lrs': list(),
            'epoch_start_timestamps': list(),
            'epoch_end_timestamps': list(),
            # ---- 新增：CCC（体积一致性相关系数），值域 [-1,1]，越接近 1 越好 ----
            'val_ccc': list(),
            # ---- 新增：HD95（95% Hausdorff Distance），单位与 spacing 一致，越小越好 ----
            'val_hd95': list(),
        }
        self.verbose = verbose
        # shut up, this logging is great

    def log(self, key, value, epoch: int):
        """
        sometimes shit gets messed up. We try to catch that here
        """
        assert key in self.my_fantastic_logging.keys() and isinstance(self.my_fantastic_logging[key], list), \
            'This function is only intended to log stuff to lists and to have one entry per epoch'

        if self.verbose: print(f'logging {key}: {value} for epoch {epoch}')

        if len(self.my_fantastic_logging[key]) < (epoch + 1):
            self.my_fantastic_logging[key].append(value)
        else:
            assert len(self.my_fantastic_logging[key]) == (epoch + 1), 'something went horribly wrong. My logging ' \
                                                                       'lists length is off by more than 1'
            print(f'maybe some logging issue!? logging {key} and {value}')
            self.my_fantastic_logging[key][epoch] = value

        # handle the ema_fg_dice special case! It is automatically logged when we add a new mean_fg_dice
        if key == 'mean_fg_dice':
            new_ema_pseudo_dice = self.my_fantastic_logging['ema_fg_dice'][epoch - 1] * 0.9 + 0.1 * value \
                if len(self.my_fantastic_logging['ema_fg_dice']) > 0 else value
            self.log('ema_fg_dice', new_ema_pseudo_dice, epoch)

    def plot_progress_png(self, output_folder):
        # we infer the epoch form our internal logging
        epoch = min([len(i) for i in self.my_fantastic_logging.values()]) - 1  # lists of epoch 0 have len 1
        sns.set(font_scale=2.5)
        # 扩展为 5 个子图（Loss/Dice、耗时、学习率、CCC、HD95）
        fig, ax_all = plt.subplots(5, 1, figsize=(30, 88))

        # 子图1：训练/验证 Loss + 伪 Dice
        ax = ax_all[0]
        ax2 = ax.twinx()
        x_values = list(range(epoch + 1))
        ax.plot(x_values, self.my_fantastic_logging['train_losses'][:epoch + 1], color='b', ls='-', label="loss_tr", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_losses'][:epoch + 1], color='r', ls='-', label="loss_val", linewidth=4)
        ax2.plot(x_values, self.my_fantastic_logging['mean_fg_dice'][:epoch + 1], color='g', ls='dotted', label="pseudo dice",
                 linewidth=3)
        ax2.plot(x_values, self.my_fantastic_logging['ema_fg_dice'][:epoch + 1], color='g', ls='-', label="pseudo dice (mov. avg.)",
                 linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("pseudo dice")
        ax.legend(loc=(0, 1))
        ax2.legend(loc=(0.2, 1))

        # 子图2：epoch 耗时
        ax = ax_all[1]
        ax.plot(x_values, [i - j for i, j in zip(self.my_fantastic_logging['epoch_end_timestamps'][:epoch + 1],
                                                 self.my_fantastic_logging['epoch_start_timestamps'])][:epoch + 1], color='b',
                ls='-', label="epoch duration", linewidth=4)
        ylim = [0] + [ax.get_ylim()[1]]
        ax.set(ylim=ylim)
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        ax.legend(loc=(0, 1))

        # 子图3：学习率
        ax = ax_all[2]
        ax.plot(x_values, self.my_fantastic_logging['lrs'][:epoch + 1], color='b', ls='-', label="learning rate", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))

        # 子图4：CCC（体积一致性相关系数）
        ax = ax_all[3]
        ccc_values = self.my_fantastic_logging['val_ccc'][:epoch + 1]
        valid_ccc = [v for v in ccc_values if v is not None and not (isinstance(v, float) and v != v)]  # 过滤 None/nan
        if len(valid_ccc) > 0:
            ax.plot(x_values[:len(ccc_values)], ccc_values, color='purple', ls='-', label="val CCC (volume)", linewidth=4)
            ax.axhline(y=0.9, color='orange', ls='--', linewidth=2, label="clinical threshold (0.9)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("CCC (volume accuracy)")
        ax.set_ylim([-1, 1])
        ax.legend(loc=(0, 1))
        ax.set_title("CCC: 体积一致性相关系数 (Concordance Correlation Coefficient)\n越接近1表示预测体积越准确")

        # 子图5：HD95（95% Hausdorff Distance）
        ax = ax_all[4]
        hd95_values = self.my_fantastic_logging['val_hd95'][:epoch + 1]
        valid_hd95 = [v for v in hd95_values if v is not None and not (isinstance(v, float) and v != v)]
        if len(valid_hd95) > 0:
            ax.plot(x_values[:len(hd95_values)], hd95_values, color='darkorange', ls='-', label="val HD95", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("HD95")
        ax.legend(loc=(0, 1))
        ax.set_title("HD95: 95% Hausdorff Distance\n越接近0表示边界距离越小")

        plt.tight_layout()

        fig.savefig(join(output_folder, "progress.png"))
        plt.close()

    def get_checkpoint(self):
        return self.my_fantastic_logging

    def load_checkpoint(self, checkpoint: dict):
        self.my_fantastic_logging = checkpoint
        # 兼容旧版 checkpoint（没有 val_ccc/val_hd95 字段）
        if 'val_ccc' not in self.my_fantastic_logging:
            self.my_fantastic_logging['val_ccc'] = []
        if 'val_hd95' not in self.my_fantastic_logging:
            self.my_fantastic_logging['val_hd95'] = []
