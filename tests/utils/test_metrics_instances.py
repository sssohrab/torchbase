import unittest
from torchbase.utils.metrics_instances import BinaryClassificationMetrics

import numpy as np
import torch


class BinaryClassificationMetricsUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metric_calculator = BinaryClassificationMetrics()

    def test_typical_usecase(self):
        ground_truth = torch.randint(0, 2, (100,))
        noise = torch.normal(0, 0.5, size=ground_truth.shape)
        predictions = 1 / (1 + torch.exp(-ground_truth - noise))

        metrics = {name: value(binary_ground_truth=ground_truth, prediction_probabilities=predictions)
                   for name, value in self.metric_calculator.get_all_metric_functionals_dict().items()}

        print(metrics)

    def test_micro_metrics_calculation(self):
        ground_truth = torch.tensor([1, 0, 1, 0, 1, 0])
        predictions = torch.tensor([1, 1, 1, 0, 0, 0])

        tp = torch.sum((predictions == 1) & (ground_truth == 1)).item()
        fp = torch.sum((predictions == 1) & (ground_truth == 0)).item()
        fn = torch.sum((predictions == 0) & (ground_truth == 1)).item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        calculated_precision = self.metric_calculator.precision_micro(binary_ground_truth=ground_truth,
                                                                      prediction_probabilities=predictions)
        calculated_recall = self.metric_calculator.recall_micro(binary_ground_truth=ground_truth,
                                                                prediction_probabilities=predictions)
        calculated_f1_score = self.metric_calculator.f1_score_micro(binary_ground_truth=ground_truth,
                                                                    prediction_probabilities=predictions)

        self.assertAlmostEqual(calculated_precision, precision, places=5)
        self.assertAlmostEqual(calculated_recall, recall, places=5)
        self.assertAlmostEqual(calculated_f1_score, f1_score, places=5)

    def test_input_type_error(self):
        ground_truth = [1, 0, 1, 0]
        predictions = np.array([1, 0, 1, 1])
        with self.assertRaises(AssertionError):
            self.metric_calculator.precision_micro(binary_ground_truth=ground_truth,
                                                   prediction_probabilities=predictions)

    def test_input_shape_mismatch_error(self):
        ground_truth = torch.Tensor([1, 0, 1, 1])
        predictions = torch.Tensor([1, 0, 1])
        with self.assertRaises(AssertionError):
            self.metric_calculator.precision_micro(binary_ground_truth=ground_truth,
                                                   prediction_probabilities=predictions)

    def test_invalid_non_probability_predictions(self):
        ground_truth = torch.tensor([1, 1, 0, 0, 0])
        predictions = torch.normal(mean=0.0, std=1.0, size=ground_truth.shape)

        with self.assertRaises(AssertionError):
            self.metric_calculator.pr_auc(binary_ground_truth=ground_truth, prediction_probabilities=predictions)


if __name__ == '__main__':
    unittest.main()
