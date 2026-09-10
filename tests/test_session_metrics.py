"""Exercise metric argument filtering through real training and validation steps."""

import tempfile
import unittest

import torch

from tests.test_session_resume import ResumeSession
from torchbase.utils.metrics import BaseMetricsClass
from torchbase.utils.session import RandomnessGeneratorStates


class IterationMetrics(BaseMetricsClass):
    @staticmethod
    def error(*, prediction: torch.Tensor, target: torch.Tensor, scale: float = 2.0) -> float:
        return (prediction.detach() - target).square().mean().item() * scale

    @staticmethod
    def target_mean(*, target: torch.Tensor) -> float:
        return target.mean().item()


class MetricSession(ResumeSession):
    def init_metrics(self):
        return [IterationMetrics(self.config_data["keyword_maps"])]

    def forward_pass(self, mini_batch):
        outputs = super().forward_pass(mini_batch)
        outputs.update(self.config_data.get("extra_outputs", {}))
        outputs["unrelated"] = 99.0
        self.last_outputs = outputs
        return outputs


class SessionMetricMappingUnitTest(unittest.TestCase):
    def test_partial_and_complete_mappings_during_training_and_validation(self):
        self.addCleanup(RandomnessGeneratorStates().apply)
        storage = tempfile.TemporaryDirectory()
        self.addCleanup(storage.cleanup)
        cases = (
            ({"output": "prediction"}, {}, 2.0),
            ({"output": "prediction"}, {"scale": 3.0}, 3.0),
            ({"output": "prediction", "factor": "scale"}, {}, 2.0),
            ({"output": "prediction", "factor": "scale"}, {"factor": 4.0}, 4.0),
            ({"output": "prediction", "target": "target", "factor": "scale"}, {"factor": 5.0}, 5.0),
        )
        for index, (mapping, extras, scale) in enumerate(cases):
            with self.subTest(mapping=mapping, extras=extras):
                config = ResumeSession.get_config(num_epochs=1)
                config["data"] = {"keyword_maps": mapping, "extra_outputs": extras}
                config["metrics"] = {"IterationMetrics": ["error", "target_mean"]}
                session = MetricSession(config, runs_parent_dir=storage.name, tag_postfix="metrics-{}".format(index))
                self.addCleanup(session.writer.close)
                batch = next(iter(session.dataloader_train))
                session.do_one_training_iteration(batch)
                self.assert_metrics(session.value_logger_train.current_values, session.last_outputs, scale)
                session.do_one_validation_iteration(batch, "valid")
                self.assert_metrics(session.value_logger_valid_dict["valid"].current_values, session.last_outputs, scale)

    def assert_metrics(self, values, outputs, scale):
        self.assertAlmostEqual(values["error"], IterationMetrics.error(
            prediction=outputs["output"], target=outputs["target"], scale=scale))
        self.assertAlmostEqual(values["target_mean"], IterationMetrics.target_mean(target=outputs["target"]))
