from torchbase.utils.metrics import BaseMetricsClass, EpochMetric

from sklearn import metrics
import numpy as np
import torch

from typing import Tuple


class BinaryClassificationMetrics(BaseMetricsClass):
    def get_epoch_metric(self, name: str) -> EpochMetric | None:
        if name in ("precision_micro", "precision_macro", "recall_micro", "recall_macro",
                    "f1_score_micro", "f1_score_macro"):
            return _BinaryClassificationEpochMetric(name)
        return None

    @staticmethod
    def _check_and_prepare_inputs(*, binary_ground_truth: torch.Tensor,
                                  prediction_probabilities: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        assert isinstance(binary_ground_truth,
                          torch.Tensor), "The `binary_ground_truth` should be a torch tensor."
        assert isinstance(prediction_probabilities,
                          torch.Tensor), "The `prediction_probabilities` should be a torch tensor."

        binary_ground_truth = binary_ground_truth.detach().cpu().numpy().flatten()
        prediction_probabilities = prediction_probabilities.detach().cpu().numpy().flatten()

        assert binary_ground_truth.shape == prediction_probabilities.shape, (
            "The `binary_ground_truth` and `prediction_probabilities` arrays should have the same size.")

        assert set(np.unique(binary_ground_truth)).issubset({0, 1}), (
            "The `binary_ground_truth` consists of more than 2 unique values and hence not appropriate for "
            "binary classification.")

        assert np.all(prediction_probabilities <= 1.0) and np.all(prediction_probabilities >= 0.0), (
            "All `prediction_probabilities` must be between 0.0 and 1.0.")

        return binary_ground_truth, prediction_probabilities

    @staticmethod
    def _round_predictions_for_point_based_metrics(prediction_probabilities: np.ndarray) -> np.ndarray:
        predicted_labels = prediction_probabilities.round()

        return predicted_labels

    def precision_micro(self, *, binary_ground_truth: torch.Tensor,
                        prediction_probabilities: torch.Tensor) -> float:
        binary_ground_truth, prediction_probabilities = self._check_and_prepare_inputs(
            binary_ground_truth=binary_ground_truth,
            prediction_probabilities=prediction_probabilities)
        predicted_labels = self._round_predictions_for_point_based_metrics(prediction_probabilities)
        value = metrics.precision_score(y_true=binary_ground_truth, y_pred=predicted_labels, average="micro")

        return value

    def precision_macro(self, *, binary_ground_truth: torch.Tensor,
                        prediction_probabilities: torch.Tensor) -> float:
        binary_ground_truth, prediction_probabilities = self._check_and_prepare_inputs(
            binary_ground_truth=binary_ground_truth,
            prediction_probabilities=prediction_probabilities)
        predicted_labels = self._round_predictions_for_point_based_metrics(prediction_probabilities)
        value = metrics.precision_score(y_true=binary_ground_truth, y_pred=predicted_labels, average="macro")

        return value

    def recall_micro(self, *, binary_ground_truth: torch.Tensor,
                     prediction_probabilities: torch.Tensor) -> float:
        binary_ground_truth, prediction_probabilities = self._check_and_prepare_inputs(
            binary_ground_truth=binary_ground_truth,
            prediction_probabilities=prediction_probabilities)
        predicted_labels = self._round_predictions_for_point_based_metrics(prediction_probabilities)
        value = metrics.recall_score(y_true=binary_ground_truth, y_pred=predicted_labels, average="micro")

        return value

    def recall_macro(self, *, binary_ground_truth: torch.Tensor,
                     prediction_probabilities: torch.Tensor) -> float:
        binary_ground_truth, prediction_probabilities = self._check_and_prepare_inputs(
            binary_ground_truth=binary_ground_truth,
            prediction_probabilities=prediction_probabilities)
        predicted_labels = self._round_predictions_for_point_based_metrics(prediction_probabilities)
        value = metrics.recall_score(y_true=binary_ground_truth, y_pred=predicted_labels, average="macro")

        return value

    def f1_score_micro(self, *, binary_ground_truth: torch.Tensor,
                       prediction_probabilities: torch.Tensor) -> float:
        binary_ground_truth, prediction_probabilities = self._check_and_prepare_inputs(
            binary_ground_truth=binary_ground_truth,
            prediction_probabilities=prediction_probabilities)
        predicted_labels = self._round_predictions_for_point_based_metrics(prediction_probabilities)
        value = metrics.f1_score(y_true=binary_ground_truth, y_pred=predicted_labels, average="micro")

        return value

    def f1_score_macro(self, *, binary_ground_truth: torch.Tensor,
                       prediction_probabilities: torch.Tensor) -> float:
        binary_ground_truth, prediction_probabilities = self._check_and_prepare_inputs(
            binary_ground_truth=binary_ground_truth,
            prediction_probabilities=prediction_probabilities)
        predicted_labels = self._round_predictions_for_point_based_metrics(prediction_probabilities)
        value = metrics.f1_score(y_true=binary_ground_truth, y_pred=predicted_labels, average="macro")

        return value

    def roc_auc(self, *, binary_ground_truth: torch.Tensor,
                prediction_probabilities: torch.Tensor) -> float:
        binary_ground_truth, prediction_probabilities = self._check_and_prepare_inputs(
            binary_ground_truth=binary_ground_truth,
            prediction_probabilities=prediction_probabilities)
        value = metrics.roc_auc_score(y_true=binary_ground_truth, y_score=prediction_probabilities)

        return value

    def pr_auc(self, *, binary_ground_truth: torch.Tensor,
               prediction_probabilities: torch.Tensor) -> float:
        binary_ground_truth, prediction_probabilities = self._check_and_prepare_inputs(
            binary_ground_truth=binary_ground_truth,
            prediction_probabilities=prediction_probabilities)
        value = metrics.average_precision_score(y_true=binary_ground_truth, y_score=prediction_probabilities)

        return value


class _BinaryClassificationEpochMetric(EpochMetric):
    """Exact flattened binary scores with four counts, not retained predictions."""

    def __init__(self, name):
        self.name = name
        self.reset()

    def update(self, *, binary_ground_truth: torch.Tensor, prediction_probabilities: torch.Tensor) -> None:
        truth, probabilities = BinaryClassificationMetrics._check_and_prepare_inputs(
            binary_ground_truth=binary_ground_truth, prediction_probabilities=prediction_probabilities)
        predicted = BinaryClassificationMetrics._round_predictions_for_point_based_metrics(probabilities)
        counts = np.bincount(2 * truth.astype(np.int64) + predicted.astype(np.int64), minlength=4)
        self.counts = [previous + int(current) for previous, current in zip(self.counts, counts)]

    def compute(self) -> float:
        tn, fp, fn, tp = self.counts
        total = sum(self.counts)
        if total == 0:
            return float("nan")
        if self.name.endswith("_micro"):
            return (tn + tp) / total
        # Match sklearn's default label set (labels present in truth or predictions)
        # and zero_division=0, including single-class epochs.
        scores = []
        for correct, false_positive, false_negative in ((tn, fn, fp), (tp, fp, fn)):
            if correct + false_positive + false_negative == 0:
                continue
            if self.name == "precision_macro":
                numerator, denominator = correct, correct + false_positive
            elif self.name == "recall_macro":
                numerator, denominator = correct, correct + false_negative
            else:
                numerator, denominator = 2 * correct, 2 * correct + false_positive + false_negative
            scores.append(numerator / denominator if denominator else 0.0)
        return sum(scores) / len(scores)

    def reset(self) -> None:
        self.counts = [0, 0, 0, 0]  # TN, FP, FN, TP

    def state_dict(self) -> dict:
        return {"name": self.name, "counts": list(self.counts)}

    def load_state_dict(self, state: dict) -> None:
        counts = state.get("counts")
        if (state.get("name") != self.name or not isinstance(counts, list) or len(counts) != 4
                or any(type(count) is not int or count < 0 for count in counts)):
            raise ValueError("Invalid or incompatible binary epoch metric state.")
        self.counts = list(counts)


class ImageReconstructionMetrics(BaseMetricsClass):
    EPS = 1e-9
    EXPECTED_MAX_PIXEL_VALUE = 1.0

    def _check_inputs(self, *, target_image: torch.Tensor,
                      output_image: torch.Tensor) -> None:

        assert isinstance(target_image, torch.Tensor) and isinstance(output_image, torch.Tensor), (
            "Both `target_image` and `output_image` should be `torch.Tensor` instances.")

        assert target_image.dtype == torch.float and output_image.dtype == torch.float, (
            "Both `target_image` and `output_image` should be tensors of floats (supposedly between 0 and {}).".format(
                self.EXPECTED_MAX_PIXEL_VALUE))

        assert target_image.shape == output_image.shape, (
            "Both `target_image` and `output_image` should have the same shapes.")

    def mse_normalized(self, *, target_image: torch.Tensor, output_image: torch.Tensor) -> float:
        self._check_inputs(target_image=target_image, output_image=output_image)

        mse = torch.nn.functional.mse_loss(target_image, output_image, reduction="mean").item()
        norm = torch.nn.functional.mse_loss(target_image, torch.zeros_like(target_image), reduction="mean").item()

        return mse / (norm + self.EPS)

    def psnr(self, *, target_image: torch.Tensor, output_image: torch.Tensor) -> float:

        self._check_inputs(target_image=target_image, output_image=output_image)

        mse = torch.nn.functional.mse_loss(output_image, target_image) + self.EPS

        psnr = 10 * torch.log10(self.EXPECTED_MAX_PIXEL_VALUE ** 2 / mse).item()

        return psnr


class ImageSegmentationMetrics(BaseMetricsClass):
    EPS = 1e-6

    @staticmethod
    def _check_inputs(*, target_binary_image: torch.Tensor,
                      output_binary_image: torch.Tensor) -> None:

        assert isinstance(target_binary_image, torch.Tensor) and isinstance(output_binary_image, torch.Tensor), (
            "Both `target_binary_image` and `output_binary_image` should be `torch.Tensor` instances.")

        assert target_binary_image.dtype == torch.bool and output_binary_image.dtype == torch.bool, (
                "Both `target_binary_image` and `output_binary_image` should be binary tensors.")

        assert target_binary_image.shape == output_binary_image.shape, (
            "Both `target_image` and `output_image` should have the same shapes.")

    def dice_score(self, *, target_binary_image: torch.Tensor,
                   output_binary_image: torch.Tensor) -> float:
        self._check_inputs(target_binary_image=target_binary_image, output_binary_image=output_binary_image)

        intersection = (
                target_binary_image & output_binary_image).sum().float()  # true positives (1s in both masks)
        target_sum = target_binary_image.sum().float()
        output_sum = output_binary_image.sum().float()

        dice = (2.0 * intersection) / (target_sum + output_sum + self.EPS)

        return dice.item()

    def iou_score(self, *, target_binary_image: torch.Tensor,
                  output_binary_image: torch.Tensor) -> float:
        self._check_inputs(target_binary_image=target_binary_image, output_binary_image=output_binary_image)

        intersection = (target_binary_image & output_binary_image).sum().float()
        union = (target_binary_image | output_binary_image).sum().float()

        iou = intersection / (union + self.EPS)
        return iou.item()
