import os
import tempfile
import unittest

import numpy as np

from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.nnUNetTrainer.trainer_attention import MyTrainer_Attention
from nnunetv2.utilities.hd95_metric import compute_hd95


class DummyNpySegReaderWriter(BaseReaderWriter):
    def __init__(self, spacing=(1.0, 1.0, 1.0)):
        self.spacing = spacing

    def read_images(self, image_fnames):
        raise NotImplementedError

    def read_seg(self, seg_fname: str):
        return np.load(seg_fname), {"spacing": self.spacing}

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        np.save(output_fname, seg)


class TestHD95Metric(unittest.TestCase):
    def test_hd95_identical_2d_is_zero(self):
        mask = np.zeros((1, 16, 16), dtype=bool)
        mask[0, 4:8, 5:9] = True
        self.assertEqual(compute_hd95(mask, mask, (999.0, 1.0, 1.0)), 0.0)

    def test_hd95_identical_3d_is_zero(self):
        mask = np.zeros((8, 8, 8), dtype=bool)
        mask[2:5, 2:5, 2:5] = True
        self.assertEqual(compute_hd95(mask, mask, (1.0, 1.0, 1.0)), 0.0)

    def test_hd95_scales_with_spacing(self):
        ref = np.zeros((16, 16), dtype=bool)
        pred = np.zeros((16, 16), dtype=bool)
        ref[4:8, 4:8] = True
        pred[5:9, 4:8] = True

        hd95_unit = compute_hd95(ref, pred, (1.0, 1.0))
        hd95_scaled = compute_hd95(ref, pred, (2.0, 1.0))
        self.assertGreater(hd95_scaled, hd95_unit)
        self.assertAlmostEqual(hd95_scaled, hd95_unit * 2.0, places=5)

    def test_hd95_empty_mask_policy(self):
        empty = np.zeros((8, 8), dtype=bool)
        full = np.zeros((8, 8), dtype=bool)
        full[2:6, 2:6] = True

        self.assertEqual(compute_hd95(empty, empty, (1.0, 1.0)), 0.0)
        self.assertAlmostEqual(
            compute_hd95(empty, full, (1.0, 2.0)),
            np.sqrt((8 * 1.0) ** 2 + (8 * 2.0) ** 2),
        )


class TestEvaluationMetrics(unittest.TestCase):
    def test_summary_contains_dice_ccc_hd95(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_dir = os.path.join(tmpdir, "gt")
            pred_dir = os.path.join(tmpdir, "pred")
            os.makedirs(gt_dir, exist_ok=True)
            os.makedirs(pred_dir, exist_ok=True)

            gt_case_1 = np.zeros((1, 8, 8, 8), dtype=np.uint8)
            gt_case_1[:, 2:5, 2:5, 2:5] = 1
            pred_case_1 = gt_case_1.copy()

            gt_case_2 = np.zeros((1, 8, 8, 8), dtype=np.uint8)
            gt_case_2[:, 1:4, 1:4, 1:4] = 1
            pred_case_2 = np.zeros((1, 8, 8, 8), dtype=np.uint8)
            pred_case_2[:, 2:5, 1:4, 1:4] = 1

            np.save(os.path.join(gt_dir, "case1.npy"), gt_case_1)
            np.save(os.path.join(pred_dir, "case1.npy"), pred_case_1)
            np.save(os.path.join(gt_dir, "case2.npy"), gt_case_2)
            np.save(os.path.join(pred_dir, "case2.npy"), pred_case_2)

            result = compute_metrics_on_folder(
                gt_dir,
                pred_dir,
                output_file=os.path.join(pred_dir, "summary.json"),
                image_reader_writer=DummyNpySegReaderWriter(),
                file_ending=".npy",
                regions_or_labels=[1],
                num_processes=1,
            )

            self.assertIn("Dice", result["metric_per_case"][0]["metrics"][1])
            self.assertIn("HD95", result["metric_per_case"][0]["metrics"][1])
            self.assertIn("CCC", result["mean"][1])
            self.assertIn("HD95", result["mean"][1])
            self.assertIn("CCC", result["foreground_mean"])
            self.assertIn("HD95", result["foreground_mean"])


class TestTrainingLogging(unittest.TestCase):
    def test_logger_plot_and_checkpoint_support_val_hd95(self):
        logger = nnUNetLogger()
        logger.log("train_losses", 1.0, 0)
        logger.log("val_losses", 0.8, 0)
        logger.log("lrs", 1e-3, 0)
        logger.log("epoch_start_timestamps", 0.0, 0)
        logger.log("epoch_end_timestamps", 1.0, 0)
        logger.log("mean_fg_dice", 0.7, 0)
        logger.log("dice_per_class_or_region", [0.7], 0)
        logger.log("val_ccc", 0.9, 0)
        logger.log("val_hd95", 1.5, 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            logger.plot_progress_png(tmpdir)
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "progress.png")))

        old_checkpoint = logger.get_checkpoint().copy()
        old_checkpoint.pop("val_hd95")
        logger.load_checkpoint(old_checkpoint)
        self.assertIn("val_hd95", logger.my_fantastic_logging)

    def test_trainer_epoch_aggregation_logs_val_hd95(self):
        trainer = MyTrainer_Attention.__new__(MyTrainer_Attention)
        trainer.is_ddp = False
        trainer.logger = nnUNetLogger()
        trainer.current_epoch = 0

        val_outputs = [
            {
                "loss": np.array(1.0),
                "tp_hard": np.array([4.0]),
                "fp_hard": np.array([1.0]),
                "fn_hard": np.array([1.0]),
                "vol_pred": np.array([5.0]),
                "vol_ref": np.array([5.0]),
                "hd95": np.array([[1.0], [2.0]]),
            },
            {
                "loss": np.array(0.5),
                "tp_hard": np.array([5.0]),
                "fp_hard": np.array([1.0]),
                "fn_hard": np.array([0.0]),
                "vol_pred": np.array([6.0]),
                "vol_ref": np.array([5.0]),
                "hd95": np.array([[3.0], [4.0]]),
            },
        ]

        trainer.on_validation_epoch_end(val_outputs)
        self.assertEqual(trainer.logger.my_fantastic_logging["val_hd95"][0], 2.5)


if __name__ == "__main__":
    unittest.main()
