"""Small CPU experiments exercising real save/load/continue behavior."""

from dataclasses import asdict
from pathlib import Path
import random
import tempfile
import unittest

from datasets import Dataset
import numpy as np
import torch

from torchbase import TrainingBaseSession
from torchbase.session import SAVED_RNG_NAME, SAVED_CONTINUATION_RNG_NAME
from torchbase.utils.data import ValidationDatasetsDict
from torchbase.utils.session import RandomnessGeneratorStates


class ResumeSession(TrainingBaseSession):
    @staticmethod
    def get_config(num_epochs=3):
        return {
            "session": {"device_name": "cpu", "num_epochs": num_epochs,
                        "mini_batch_size": 3, "learning_rate": 0.01},
            "data": {}, "metrics": {}, "network": {"architecture": "Sequential"},
        }

    def init_datasets(self):
        # Reconstructing these data requires the initialization RNG, not the
        # continuation RNG saved after an epoch.
        inputs = torch.rand(8, 2) + np.random.random() + random.random()
        dataset = Dataset.from_dict({"inputs": inputs, "target": inputs.sum(1, keepdim=True)})
        dataset.set_format("torch")
        return dataset, ValidationDatasetsDict(
            datasets=(dataset.select(range(5)), dataset.select(range(3))),
            only_for_demo=(False, True), names=("valid", "demo"),
        )

    def init_network(self):
        return torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Dropout(0.25))

    def init_metrics(self):
        return None

    def forward_pass(self, mini_batch):
        noise = random.random() * 0.01 + np.random.random() * 0.01
        return {"output": self.network(mini_batch["inputs"] + noise), "target": mini_batch["target"]}

    def loss_function(self, *, output, target):
        return torch.nn.functional.mse_loss(output, target)

    @staticmethod
    def print(report, end=None):
        pass


class TrainingBaseSessionResumeUnitTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(RandomnessGeneratorStates().apply)
        self.storage = tempfile.TemporaryDirectory()
        self.addCleanup(self.storage.cleanup)
        self.seed()

    @staticmethod
    def seed():
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

    def make_session(self, *, epochs=3, source=None, fresh=True, tag="run"):
        session = ResumeSession(
            config=ResumeSession.get_config(epochs), runs_parent_dir=self.storage.name,
            source_run_dir_tag=source, create_run_dir_afresh=fresh,
            tag_postfix=tag if fresh else None,
        )
        self.addCleanup(session.writer.close)
        return session

    def resume(self, session, *, epochs=3):
        return self.make_session(epochs=epochs, source=Path(session.run_dir).name, fresh=False)

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for actual_item, expected_item in zip(actual, expected):
                self.assert_nested_equal(actual_item, expected_item)
        else:
            self.assertEqual(actual, expected)

    def assert_loggers_equal(self, actual, expected):
        self.assertEqual(actual.names, expected.names)
        self.assertEqual(actual.current_values, expected.current_values)
        self.assertEqual(actual.average_of_epoch, expected.average_of_epoch)
        self.assertEqual(actual.average_overall, expected.average_overall)

    def assert_states_equal(self, actual, expected):
        self.assertEqual(asdict(actual.progress_train), asdict(expected.progress_train))
        self.assert_loggers_equal(actual.value_logger_train, expected.value_logger_train)
        self.assertIs(actual.value_logger_train.progress_manager, actual.progress_train)
        for name in expected.datasets_valid_dict.names:
            self.assertEqual(asdict(actual.progress_valid_dict[name]), asdict(expected.progress_valid_dict[name]))
            self.assert_loggers_equal(actual.value_logger_valid_dict[name], expected.value_logger_valid_dict[name])
            self.assertIs(actual.value_logger_valid_dict[name].progress_manager, actual.progress_valid_dict[name])
        self.assert_nested_equal(actual.network.state_dict(), expected.network.state_dict())
        self.assert_nested_equal(actual.optimizer.state_dict(), expected.optimizer.state_dict())

    def test_resume_restores_progress_loggers_model_and_optimizer(self):
        original = self.make_session(epochs=1)
        original.train()
        resumed = self.resume(original)
        self.assert_states_equal(resumed, original)
        self.assertEqual(resumed.dataset_train.data, original.dataset_train.data)

    def test_resumed_training_matches_uninterrupted_training(self):
        uninterrupted = self.make_session(tag="uninterrupted")
        uninterrupted.train()
        expected_next_random = (torch.rand(3).tolist(), np.random.rand(3).tolist(), random.random())

        self.seed()
        interrupted = self.make_session(epochs=1, tag="interrupted")
        initialization_rng_path = Path(interrupted.run_dir, "states", SAVED_RNG_NAME)
        initialization_rng = initialization_rng_path.read_bytes()
        interrupted.train()
        self.assertEqual(initialization_rng_path.read_bytes(), initialization_rng)

        # Unrelated work must not affect the resumed experiment.
        torch.rand(20)
        np.random.rand(20)
        random.random()
        resumed = self.resume(interrupted)
        resumed.train()

        self.assert_states_equal(resumed, uninterrupted)
        self.assertEqual((torch.rand(3).tolist(), np.random.rand(3).tolist(), random.random()), expected_next_random)

    def test_completed_run_does_not_repeat_epochs(self):
        original = self.make_session(epochs=1)
        original.train()
        resumed = self.resume(original, epochs=1)
        resumed.train()
        self.assert_states_equal(resumed, original)

    def test_new_run_with_existing_weights_does_not_restore_training_state(self):
        original = self.make_session(epochs=1)
        original.train()
        warm_start = self.make_session(source=Path(original.run_dir).name, tag="warm-start")
        self.assertNotEqual(warm_start.run_dir, original.run_dir)
        self.assert_nested_equal(warm_start.network.state_dict(), original.network.state_dict())
        self.assertEqual(warm_start.optimizer.state_dict()["state"], {})
        self.assertEqual(warm_start.progress_train.epoch, 0)
        self.assertEqual(warm_start.value_logger_train.average_overall, {"loss": 0.0})
        for name in warm_start.datasets_valid_dict.names:
            self.assertEqual(warm_start.progress_valid_dict[name].epoch, 0)
        self.assertEqual(warm_start.best_validation_loss_dict["valid"][0], float("inf"))

    def test_missing_progress_file_is_not_silently_reset(self):
        original = self.make_session(epochs=1)
        original.train()
        # Only this test's disposable checkpoint is modified.
        Path(original.run_dir, "states", "progress_manager_train.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "progress_manager_train"):
            self.resume(original)

    def test_legacy_checkpoint_restores_available_state_with_warning(self):
        original = self.make_session(epochs=1)
        original.train()
        Path(original.run_dir, "states", SAVED_CONTINUATION_RNG_NAME).unlink()
        with self.assertWarnsRegex(RuntimeWarning, "no continuation RNG state"):
            resumed = self.resume(original)
        self.assert_states_equal(resumed, original)

    def test_partial_validation_epoch_cannot_resume(self):
        original = self.make_session(epochs=1)
        original.train()
        original.do_one_validation_iteration(next(iter(original.dataloader_valid_dict["demo"])), "demo")
        original.save_progress_and_log_states_for_valid_set("demo")
        with self.assertRaisesRegex(ValueError, "completed.*epoch"):
            self.resume(original)

    def test_partial_training_epoch_cannot_resume(self):
        original = self.make_session(epochs=1)
        original.train()
        original.do_one_training_iteration(next(iter(original.dataloader_train)))
        original.save_training_states()
        with self.assertRaisesRegex(ValueError, "completed.*epoch"):
            self.resume(original)

    def test_mismatched_training_and_validation_epochs_cannot_resume(self):
        original = self.make_session(epochs=1)
        original.train()
        original.do_one_training_epoch()
        original.save_training_states()
        with self.assertRaisesRegex(ValueError, "completed.*epoch"):
            self.resume(original)


if __name__ == "__main__":
    unittest.main()
