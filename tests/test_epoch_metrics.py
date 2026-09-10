"""Dataset scores, batch-score averages, and recovery must remain distinct."""

from copy import deepcopy
from pathlib import Path
import math
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from datasets import Dataset
from sklearn import metrics as reference
from torchdata.stateful_dataloader import StatefulDataLoader

from tests.test_session_resume import ResumeSession
from torchbase.utils.data import ValidationDatasetsDict
from torchbase.utils.logger import LoggableParams, ProgressManager, ValuesLogger
from torchbase.utils.metrics import BaseMetricsClass, EpochMetric
from torchbase.utils.metrics_instances import BinaryClassificationMetrics
from torchbase.utils.session import RandomnessGeneratorStates, is_custom_scalar_logging_layout_valid


NAMES = ["precision_micro", "precision_macro", "recall_micro", "recall_macro", "f1_score_micro", "f1_score_macro"]


def reference_score(name, truth, probabilities):
    scoring = {"precision": reference.precision_score, "recall": reference.recall_score,
               "f1_score": reference.f1_score}
    kind, average = name.rsplit("_", 1)
    return scoring[kind](truth, np.round(probabilities), average=average, zero_division=0)


class BinaryEpochMetricUnitTest(unittest.TestCase):
    def test_uneven_batches_match_whole_dataset_reference_not_batch_averages(self):
        truth = torch.tensor([0, 1, 1, 0, 1])
        probabilities = torch.tensor([0.1, 0.6, 0.2, 0.7, 0.8])
        group = BinaryClassificationMetrics({"target": "binary_ground_truth", "output": "prediction_probabilities"})
        progress = ProgressManager()
        logger = ValuesLogger(NAMES, progress, group.get_epoch_metrics(NAMES))
        functions = LoggableParams(group.get_metrics(NAMES))
        for section in (slice(0, 3), slice(3, 5)):
            inputs = {"target": truth[section], "output": probabilities[section]}
            progress.increment_iter(len(inputs["target"]))
            logger.update(functions(**inputs), metric_inputs=inputs)
        for name in NAMES:
            self.assertAlmostEqual(logger.epoch_values[name], reference_score(name, truth, probabilities))
        for name in ("precision_macro", "recall_macro", "f1_score_macro"):
            self.assertNotAlmostEqual(logger.epoch_values[name], logger.average_of_epoch[name])
        for metric in logger.epoch_metrics.values():
            self.assertEqual(metric.state_dict()["counts"], [1, 1, 1, 2])
            self.assertEqual(set(metric.state_dict()), {"name", "counts"})

    def test_single_class_zero_denominators_and_rounding_match_reference(self):
        for truth, probabilities in (([0, 0, 0], [0.1, 0.2, 0.5]), ([1, 1], [0.8, 0.9]),
                                     ([0, 1, 1], [0.1, 0.2, 0.5]), ([0, 0], [0.9, 0.8]),
                                     ([1, 1], [0.1, 0.2])):
            accumulators = BinaryClassificationMetrics().get_epoch_metrics(NAMES)
            for name, metric in accumulators.items():
                with self.subTest(name=name, truth=truth, probabilities=probabilities):
                    for label, score in zip(truth, probabilities):
                        metric.update(binary_ground_truth=torch.tensor([label]),
                                      prediction_probabilities=torch.tensor([score]))
                    self.assertAlmostEqual(metric.compute(), reference_score(name, truth, probabilities))

    def test_bounded_state_round_trip_and_reset(self):
        group = BinaryClassificationMetrics()
        for name in NAMES:
            with self.subTest(name=name):
                metric = group.get_epoch_metric(name)
                self.assertTrue(math.isnan(metric.compute()))
                for _ in range(100):
                    metric.update(binary_ground_truth=torch.tensor([0, 1]),
                                  prediction_probabilities=torch.tensor([0.8, 0.9]))
                state = metric.state_dict()
                self.assertEqual(state["counts"], [0, 100, 0, 100])
                restored = group.get_epoch_metric(name)
                restored.load_state_dict(state)
                state["counts"][0] = 999
                self.assertEqual(restored.compute(), metric.compute())
                restored.reset()
                self.assertTrue(math.isnan(restored.compute()))
                self.assertEqual(metric.state_dict()["counts"], [0, 100, 0, 100])

    def test_invalid_state_and_inputs_are_rejected(self):
        metric = BinaryClassificationMetrics().get_epoch_metric("f1_score_macro")
        for state in ({}, {"name": "precision_macro", "counts": [0, 0, 0, 0]},
                      {"name": "f1_score_macro", "counts": [0, 0, 0]},
                      {"name": "f1_score_macro", "counts": [0, -1, 0, 0]},
                      {"name": "f1_score_macro", "counts": [0, True, 0, 0]}):
            with self.subTest(state=state), self.assertRaises(ValueError):
                metric.load_state_dict(state)
        with self.assertRaises(AssertionError):
            metric.update(binary_ground_truth=torch.tensor([0, 1]),
                          prediction_probabilities=torch.tensor([0.1, 1.1]))
        self.assertEqual(metric.state_dict()["counts"], [0, 0, 0, 0])

    def test_auc_is_explicitly_only_a_batch_statistic_without_prediction_storage(self):
        group = BinaryClassificationMetrics()
        self.assertEqual(group.get_epoch_metrics(["roc_auc", "pr_auc"]), {})
        truth = torch.tensor([0, 1, 1, 0, 1])
        scores = torch.tensor([0.1, 0.2, 0.3, 0.8, 0.9])
        progress = ProgressManager()
        logger = ValuesLogger(["roc_auc"], progress, group.get_epoch_metrics(["roc_auc"]))
        for section in (slice(0, 3), slice(3, 5)):
            progress.increment_iter(len(truth[section]))
            logger.update({"roc_auc": float(reference.roc_auc_score(truth[section], scores[section]))})
        self.assertEqual(logger.epoch_values, {})
        self.assertEqual(logger.average_of_epoch["roc_auc"], 1.0)
        self.assertNotAlmostEqual(logger.average_of_epoch["roc_auc"], reference.roc_auc_score(truth, scores))
        self.assertEqual(logger.state_dict()["epoch_metrics"], {})


class SumMetric(EpochMetric):
    def __init__(self):
        self.reset()

    def update(self, *, value: torch.Tensor, scale: float = 2.0):
        self.total += value.detach().sum().item() * scale

    def compute(self):
        return self.total

    def reset(self):
        self.total = 0.0

    def state_dict(self):
        return {"total": self.total}

    def load_state_dict(self, state):
        self.total = state["total"]


class SumMetrics(BaseMetricsClass):
    @staticmethod
    def score(*, value: torch.Tensor, scale: float = 2.0) -> float:
        return value.sum().item() * scale

    def get_epoch_metric(self, name):
        return SumMetric()


class EpochMetricContractUnitTest(unittest.TestCase):
    def make_logger(self):
        return ValuesLogger(["score"], ProgressManager(),
                            SumMetrics({"prediction": "value"}).get_epoch_metrics(["score"]))

    def test_custom_metric_mapping_defaults_serialization_and_reset(self):
        logger = self.make_logger()
        for value, scale in (([1.0, 2.0], None), ([3.0], 4.0)):
            inputs = {"prediction": torch.tensor(value), "irrelevant": "ignored"}
            if scale is not None:
                inputs["scale"] = scale
            logger.progress_manager.increment_iter(len(value))
            logger.update({"score": 0.0}, metric_inputs=inputs)
        self.assertEqual(logger.epoch_values, {"score": 18.0})
        state = logger.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "logger.json")
            logger.serialize_to_disk(path)
            restored = self.make_logger()
            restored.set_state_values_from_disk(path)
        self.assertEqual(restored.epoch_values, logger.epoch_values)
        logger.reset()
        self.assertEqual(logger.epoch_values, {"score": 0.0})
        self.assertEqual(logger.progress_manager.samples_total, 0)
        self.assertEqual(state["epoch_metrics"]["score"]["state"], {"total": 18.0})

    def test_missing_or_changed_accumulators_are_rejected_on_recovery(self):
        logger = self.make_logger()
        state = logger.state_dict()
        for corrupt in ({key: value for key, value in state.items() if key != "epoch_metrics"},
                        {**state, "epoch_metrics": {}},
                        {**state, "epoch_metrics": {"score": {"type": "other", "state": {}}}}):
            with self.subTest(corrupt=corrupt), self.assertRaisesRegex(RuntimeError, "epoch metric"):
                logger.load_state_dict(corrupt)

    def test_invalid_registration_and_missing_inputs_are_rejected(self):
        for metrics in ({"missing": SumMetric()}, {"score": object()}):
            with self.subTest(metrics=metrics), self.assertRaises(ValueError):
                ValuesLogger(["score"], ProgressManager(), metrics)
        logger = self.make_logger()
        with self.assertRaisesRegex(ValueError, "metric_inputs"):
            logger.update({"score": 1.0})
        with patch.object(logger.epoch_metrics["score"], "compute", return_value=1), self.assertRaises(TypeError):
            _ = logger.epoch_values
        with patch.object(SumMetrics, "get_epoch_metric", return_value=object()), self.assertRaises(TypeError):
            SumMetrics().get_epoch_metrics(["score"])
        metric = SumMetric()
        metric.update = lambda value: None
        with patch.object(SumMetrics, "get_epoch_metric", return_value=metric), self.assertRaises(TypeError):
            SumMetrics().get_epoch_metrics(["score"])


class ClassificationSession(ResumeSession):
    def init_datasets(self):
        dataset = Dataset.from_dict({"inputs": [[0.0, 1.0]] * 5,
                                     "target": [0.0, 1.0, 1.0, 0.0, 1.0],
                                     "probability": [0.1, 0.6, 0.2, 0.7, 0.8]})
        dataset.set_format("torch")
        return dataset, ValidationDatasetsDict((dataset, dataset.select([0, 1, 3])), (False, True), ("valid", "demo"))

    def init_dataloaders(self):
        def loader(dataset):
            return StatefulDataLoader(dataset, batch_size=2, shuffle=False,
                                      generator=torch.Generator().manual_seed(42))
        return loader(self.dataset_train), {name: loader(dataset) for name, dataset in
                                            zip(self.datasets_valid_dict.names, self.datasets_valid_dict.datasets)}

    def init_metrics(self):
        return [BinaryClassificationMetrics({"target": "binary_ground_truth", "output": "prediction_probabilities"})]

    def forward_pass(self, batch):
        # Keep scores fixed for reference comparisons while exercising real backward/optimizer steps.
        return {"output": batch["probability"] + self.network(batch["inputs"]).sum() * 0,
                "target": batch["target"]}


class EpochMetricSessionUnitTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(RandomnessGeneratorStates().apply)
        self.storage = tempfile.TemporaryDirectory()
        self.addCleanup(self.storage.cleanup)

    def make_session(self, tag, source=None, epochs=2):
        config = ResumeSession.get_config(epochs)
        config["session"]["checkpoint_interval"] = 1
        config["metrics"] = {"BinaryClassificationMetrics": NAMES}
        session = ClassificationSession(config, runs_parent_dir=self.storage.name, tag_postfix=tag if source is None else None,
                                        create_run_dir_afresh=source is None, source_run_dir_tag=source)
        self.addCleanup(session.writer.close)
        return session

    def assert_reference(self, session):
        for dataset, logger in [(session.dataset_train, session.value_logger_train),
                                *zip(session.datasets_valid_dict.datasets, session.value_logger_valid_dict.values())]:
            for name in NAMES:
                self.assertAlmostEqual(logger.epoch_values[name], reference_score(
                    name, dataset[:]["target"], dataset[:]["probability"]))
                self.assertEqual(sum(logger.epoch_metrics[name].state_dict()["counts"]), len(dataset))

    def test_training_validation_and_demo_reset_independently_each_epoch(self):
        session = self.make_session("reset")
        with patch.object(session.writer, "add_scalar") as write:
            session.train()
        self.assert_reference(session)
        tags = {call.args[0] for call in write.call_args_list}
        self.assertIn("training/f1_score_macro/epochs", tags)
        self.assertIn("training/f1_score_macro/batch_means", tags)
        self.assertIn("validation-demo/f1_score_macro/epochs", tags)
        self.assertIn("training/loss/epochs", tags)
        self.assertNotIn("training/loss/batch_means", tags)
        self.assertEqual([call.args[2] for call in write.call_args_list
                          if call.args[0] == "training/f1_score_macro/epochs"], [1, 2])
        final_scores = {call.args[0]: call.args[1] for call in write.call_args_list}
        self.assertEqual(final_scores["training/f1_score_macro/epochs"], session.value_logger_train.epoch_values["f1_score_macro"])
        others = [deepcopy(logger.state_dict()) for logger in session.value_logger_valid_dict.values()]
        overall = dict(session.value_logger_train.average_overall)
        session.value_logger_train.reset_epoch()
        self.assertTrue(math.isnan(session.value_logger_train.epoch_values["f1_score_macro"]))
        self.assertEqual(session.value_logger_train.average_overall, overall)
        self.assertEqual([logger.state_dict() for logger in session.value_logger_valid_dict.values()], others)
        session.value_logger_valid_dict["valid"].reset_epoch()
        self.assertEqual(session.value_logger_valid_dict["demo"].state_dict(), others[1])

    def test_interruption_after_update_before_save_restores_all_accumulators(self):
        uninterrupted = self.make_session("uninterrupted")
        uninterrupted.train()
        expected = [logger.state_dict() for logger in
                    [uninterrupted.value_logger_train, *uninterrupted.value_logger_valid_dict.values()]]
        for phase in ("training", "valid", "demo"):
            with self.subTest(phase=phase):
                session = self.make_session("interrupted-" + phase)
                save = session._save_checkpoint

                def fail_after_update(**kwargs):
                    active = session._phase == "training" if phase == "training" else (
                        session._phase == "validation" and session.datasets_valid_dict.names[session._validation_index] == phase)
                    progress = session.progress_train if phase == "training" else session.progress_valid_dict[phase]
                    if active and progress.iter_current_epoch == 2:
                        raise RuntimeError("interrupted before checkpoint commit")
                    save(**kwargs)

                with patch.object(session, "_save_checkpoint", side_effect=fail_after_update), self.assertRaisesRegex(RuntimeError, "interrupted"):
                    session.train()
                checkpoint = torch.load(Path(session.run_dir, "states/checkpoint.pth"), weights_only=True)
                saved_values = checkpoint["values_logger_train"] if phase == "training" else checkpoint["validation"][phase]["values_logger"]
                self.assertEqual(sum(saved_values["epoch_metrics"]["f1_score_macro"]["state"]["counts"]), 2)
                recovered = self.make_session("recovered", source=Path(session.run_dir).name)
                recovered_logger = recovered.value_logger_train if phase == "training" else recovered.value_logger_valid_dict[phase]
                self.assertEqual(recovered_logger.state_dict(), saved_values)
                recovered.train()
                self.assert_reference(recovered)
                self.assertEqual([logger.state_dict() for logger in
                                  [recovered.value_logger_train, *recovered.value_logger_valid_dict.values()]], expected)

    def test_stateless_metric_tags_and_custom_layout_use_batch_means(self):
        session = self.make_session("tags")
        logger = ValuesLogger(["loss", "roc_auc"], ProgressManager())
        logger.average_of_epoch = {"loss": 0.5, "roc_auc": 0.75}
        with patch.object(session.writer, "add_scalar") as write:
            session._write_epoch_metrics(logger, "training", 1)
        self.assertEqual({call.args[0] for call in write.call_args_list}, {"training/loss/epochs", "training/roc_auc/batch_means"})
        self.assertTrue(is_custom_scalar_logging_layout_valid(
            {"AUC": ["Multiline", ["training/roc_auc/batch_means"]]}, ("valid",), ("roc_auc",)))

    def test_last_batch_checkpoint_recovery_writes_the_completed_epoch_summary(self):
        for prefix in ("training", "validation-valid", "validation-demo"):
            with self.subTest(prefix=prefix):
                session = self.make_session("summary-" + prefix, epochs=1)
                write = session._write_epoch_metrics

                def interrupt_summary(logger, current_prefix, epoch):
                    if current_prefix == prefix:
                        raise RuntimeError("interrupted before summary")
                    write(logger, current_prefix, epoch)

                with patch.object(session, "_write_epoch_metrics", side_effect=interrupt_summary), \
                        self.assertRaisesRegex(RuntimeError, "interrupted before summary"):
                    session.train()
                recovered = self.make_session("recover-summary", source=Path(session.run_dir).name, epochs=1)
                with patch.object(recovered.writer, "add_scalar") as log:
                    recovered.train()
                self.assert_reference(recovered)
                calls = [call for call in log.call_args_list if call.args[0] == prefix + "/f1_score_macro/epochs"]
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0].args[2], 1)

    def test_old_checkpoint_format_is_rejected(self):
        session = self.make_session("old", epochs=1)
        session.train()
        path = Path(session.run_dir, "states/checkpoint.pth")
        checkpoint = torch.load(path, weights_only=True)
        checkpoint["format_version"] = 2
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "Unsupported checkpoint format"):
            self.make_session("recover-old", source=Path(session.run_dir).name)

    def test_factory_must_not_share_state_between_metrics_or_datasets(self):
        shared = BinaryClassificationMetrics().get_epoch_metric("f1_score_macro")
        with patch.object(BinaryClassificationMetrics, "get_epoch_metric", return_value=shared), \
                self.assertRaisesRegex(ValueError, "fresh, independent"):
            self.make_session("shared")

    def test_missing_accumulator_state_cannot_silently_resume(self):
        session = self.make_session("missing", epochs=1)
        session.train()
        path = Path(session.run_dir, "states/checkpoint.pth")
        checkpoint = torch.load(path, weights_only=True)
        checkpoint["validation"]["demo"]["values_logger"]["epoch_metrics"].pop("f1_score_macro")
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(RuntimeError, "epoch metrics"):
            self.make_session("recover-missing", source=Path(session.run_dir).name)
