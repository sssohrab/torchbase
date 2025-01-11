from torchbase.utils.metrics import BaseMetricsClass

from sklearn import metrics
import numpy as np
import torch

from typing import Tuple


class BinaryClassificationMetrics(BaseMetricsClass):
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
