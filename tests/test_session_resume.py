"""Small CPU experiments exercising real save/load/continue behavior."""

from dataclasses import asdict
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset
import numpy as np
import torch

from torchbase import TrainingBaseSession
from torchbase.session import SAVED_RNG_NAME, SAVED_CONTINUATION_RNG_NAME, SAVED_CHECKPOINT_NAME
from torchbase.utils.checkpoint import atomic_torch_save
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


class SelectionSession(ResumeSession):
    """Controlled losses expose independent records versus joint model selection."""

    def init_datasets(self):
        dataset, _ = super().init_datasets()
        return dataset, ValidationDatasetsDict(
            datasets=(dataset, dataset, dataset), only_for_demo=(False, False, True),
            names=("first", "second", "demo"),
        )

    def do_one_training_epoch(self):
        super().do_one_training_epoch()
        with torch.no_grad():
            for parameter in self.network.parameters():
                parameter.fill_(self.progress_train.epoch)

    def do_one_validation_epoch(self, name):
        super().do_one_validation_epoch(name)
        losses = {"first": [3.0, 2.0, 2.5, 1.0, 1.0], "second": [4.0, 5.0, 3.0, 2.0, 2.0],
                  "demo": [1.0, 2.0, 3.0, 4.0, 5.0]}
        self.value_logger_valid_dict[name].average_of_epoch["loss"] = losses[name][self.progress_train.epoch - 1]


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

    def make_session(self, *, epochs=3, source=None, fresh=True, tag="run", session_class=ResumeSession):
        session = session_class(
            config=ResumeSession.get_config(epochs), runs_parent_dir=self.storage.name,
            source_run_dir_tag=source, create_run_dir_afresh=fresh,
            tag_postfix=tag if fresh else None,
        )
        self.addCleanup(session.writer.close)
        return session

    def resume(self, session, *, epochs=3):
        return self.make_session(epochs=epochs, source=Path(session.run_dir).name, fresh=False,
                                 session_class=type(session))

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
        self.assertEqual(actual.best_validation_loss_dict, expected.best_validation_loss_dict)
        self.assertEqual(asdict(actual.progress_train), asdict(expected.progress_train))
        self.assert_loggers_equal(actual.value_logger_train, expected.value_logger_train)
        self.assertIs(actual.value_logger_train.progress_manager, actual.progress_train)
        for name in expected.datasets_valid_dict.names:
            self.assertEqual(asdict(actual.progress_valid_dict[name]), asdict(expected.progress_valid_dict[name]))
            self.assert_loggers_equal(actual.value_logger_valid_dict[name], expected.value_logger_valid_dict[name])
            self.assertIs(actual.value_logger_valid_dict[name].progress_manager, actual.progress_valid_dict[name])
        self.assert_nested_equal(actual.network.state_dict(), expected.network.state_dict())
        self.assert_nested_equal(actual.optimizer.state_dict(), expected.optimizer.state_dict())

    @staticmethod
    def make_legacy_checkpoint(session):
        session.save_training_states()
        for name in session.datasets_valid_dict.names:
            session.save_progress_and_log_states_for_valid_set(name)
        # Only this test's disposable checkpoint is removed.
        Path(session.run_dir, "states", SAVED_CHECKPOINT_NAME).unlink()

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
        self.make_legacy_checkpoint(original)
        # Only this test's disposable checkpoint is modified.
        Path(original.run_dir, "states", "progress_manager_train.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "progress_manager_train"):
            self.resume(original)

    def test_legacy_checkpoint_restores_available_state_with_warning(self):
        original = self.make_session(epochs=1)
        original.train()
        self.make_legacy_checkpoint(original)
        Path(original.run_dir, "states", SAVED_CONTINUATION_RNG_NAME).unlink()
        with self.assertWarnsRegex(RuntimeWarning, "no continuation RNG state"):
            resumed = self.resume(original)
        self.assert_states_equal(resumed, original)

    def test_partial_validation_epoch_cannot_resume(self):
        original = self.make_session(epochs=1)
        original.train()
        self.make_legacy_checkpoint(original)
        original.do_one_validation_iteration(next(iter(original.dataloader_valid_dict["demo"])), "demo")
        original.save_progress_and_log_states_for_valid_set("demo")
        with self.assertRaisesRegex(ValueError, "completed.*epoch"):
            self.resume(original)

    def test_partial_training_epoch_cannot_resume(self):
        original = self.make_session(epochs=1)
        original.train()
        self.make_legacy_checkpoint(original)
        original.do_one_training_iteration(next(iter(original.dataloader_train)))
        original.save_training_states()
        with self.assertRaisesRegex(ValueError, "completed.*epoch"):
            self.resume(original)

    def test_mismatched_training_and_validation_epochs_cannot_resume(self):
        original = self.make_session(epochs=1)
        original.train()
        self.make_legacy_checkpoint(original)
        original.do_one_training_epoch()
        original.save_training_states()
        with self.assertRaisesRegex(ValueError, "completed.*epoch"):
            self.resume(original)

    def test_interrupted_training_or_validation_replays_from_completed_epoch(self):
        for phase in ("training", "valid", "demo"):
            with self.subTest(phase=phase):
                self.seed()
                expected = self.make_session(tag="expected-" + phase)
                expected.train()
                expected_random = RandomnessGeneratorStates().to_dict()

                self.seed()
                original = self.make_session(epochs=1, tag="interrupted-" + phase)
                original.train()
                resumed = self.resume(original)
                checkpoint_path = Path(original.run_dir, "states", SAVED_CHECKPOINT_NAME)
                before = checkpoint_path.read_bytes()
                method_name = ("do_one_training_iteration" if phase == "training"
                               else "do_one_validation_iteration")
                method = getattr(resumed, method_name)

                def interrupt(*args):
                    method(*args)
                    if phase == "training":
                        resumed.save_training_states()
                    else:
                        if args[1] != phase:
                            return
                        resumed.save_progress_and_log_states_for_valid_set(phase)
                    raise KeyboardInterrupt("simulated interruption")

                with patch.object(resumed, method_name, side_effect=interrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        resumed()
                self.assertEqual(checkpoint_path.read_bytes(), before)
                recovered = self.resume(original)
                self.assertEqual(recovered.progress_train.epoch, 1)
                recovered.train()
                self.assert_states_equal(recovered, expected)
                self.assertEqual(RandomnessGeneratorStates().to_dict(), expected_random)

    def test_interrupted_first_epoch_can_restart(self):
        original = self.make_session()
        method = original.do_one_training_iteration

        def interrupt(batch):
            method(batch)
            raise KeyboardInterrupt()

        with patch.object(original, "do_one_training_iteration", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                original()
        recovered = self.resume(original)
        self.assertEqual(recovered.progress_train.epoch, 0)
        self.assertEqual(recovered.optimizer.state_dict()["state"], {})
        self.assertFalse(Path(original.run_dir, "network.pth").exists())
        recovered.train()

        self.seed()
        expected = self.make_session(tag="expected")
        expected.train()
        self.assert_states_equal(recovered, expected)

    def test_failed_checkpoint_write_keeps_previous_recovery_point(self):
        original = self.make_session(epochs=1)
        original.train()
        checkpoint_path = Path(original.run_dir, "states", SAVED_CHECKPOINT_NAME)
        before = checkpoint_path.read_bytes()
        resumed = self.resume(original)

        def fail_save(state, file):
            file.write(b"incomplete checkpoint")
            raise OSError("simulated disk failure")

        with patch("torchbase.utils.checkpoint.torch.save", side_effect=fail_save):
            with self.assertRaisesRegex(OSError, "disk failure"):
                resumed()
        self.assertEqual(checkpoint_path.read_bytes(), before)
        self.assertEqual(list(checkpoint_path.parent.glob(".checkpoint-*.tmp")), [])
        self.assert_states_equal(self.resume(original), original)

    def test_failed_replace_keeps_previous_recovery_point(self):
        original = self.make_session(epochs=1)
        original.train()
        checkpoint_path = Path(original.run_dir, "states", SAVED_CHECKPOINT_NAME)
        before = checkpoint_path.read_bytes()
        with patch("torchbase.utils.checkpoint.os.replace", side_effect=OSError("replace failure")):
            with self.assertRaisesRegex(OSError, "replace failure"):
                original._save_checkpoint()
        self.assertEqual(checkpoint_path.read_bytes(), before)
        self.assertEqual(list(checkpoint_path.parent.glob(".checkpoint-*.tmp")), [])

    def test_best_export_is_repaired_after_checkpoint_commit(self):
        original = self.make_session(epochs=1)

        def fail_export(state, path):
            if Path(path).parent == Path(original.run_dir):
                raise OSError("export failure")
            atomic_torch_save(state, path)

        with patch("torchbase.session.atomic_torch_save", side_effect=fail_export):
            with self.assertRaisesRegex(OSError, "export failure"):
                original()
        self.assertFalse(Path(original.run_dir, "network.pth").exists())
        recovered = self.resume(original)
        self.assert_states_equal(recovered, original)
        self.assertEqual(recovered.best_model_epoch, 0)
        self.assert_nested_equal(torch.load(Path(original.run_dir, "network.pth"), weights_only=True),
                                 original.network.state_dict())

    def test_invalid_primary_checkpoint_does_not_fall_back_to_legacy_files(self):
        original = self.make_session(epochs=1)
        original.train()
        original.save_training_states()
        for name in original.datasets_valid_dict.names:
            original.save_progress_and_log_states_for_valid_set(name)
        checkpoint_path = Path(original.run_dir, "states", SAVED_CHECKPOINT_NAME)
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        for corrupt in ({}, {**checkpoint, "format_version": 999}):
            with self.subTest(version=corrupt.get("format_version")):
                torch.save(corrupt, checkpoint_path)
                with self.assertRaisesRegex(ValueError, "checkpoint"):
                    self.resume(original)

    def test_partial_state_cannot_replace_completed_checkpoint(self):
        original = self.make_session(epochs=1)
        original.train()
        checkpoint_path = Path(original.run_dir, "states", SAVED_CHECKPOINT_NAME)
        before = checkpoint_path.read_bytes()
        original.do_one_training_iteration(next(iter(original.dataloader_train)))
        with self.assertRaisesRegex(ValueError, "completed.*epoch"):
            original._save_checkpoint()
        self.assertEqual(checkpoint_path.read_bytes(), before)

    def test_selection_metadata_and_best_weights_survive_reload(self):
        original = self.make_session(session_class=SelectionSession)
        original.train()
        self.assertEqual(original.best_validation_loss_dict, {"first": (2.0, 1), "second": (3.0, 2)})
        self.assertEqual(original.best_model_epoch, 0)
        recovered = self.resume(original, epochs=5)
        self.assert_states_equal(recovered, original)
        self.assertEqual(recovered.best_model_epoch, 0)
        checkpoint = torch.load(Path(original.run_dir, "states", SAVED_CHECKPOINT_NAME), weights_only=True)
        self.assertEqual(checkpoint["best_validation_loss_dict"], original.best_validation_loss_dict)
        for weight in checkpoint["best_model"].values():
            self.assertTrue(torch.equal(weight, torch.ones_like(weight)))
        for weight in recovered.network.state_dict().values():
            self.assertTrue(torch.equal(weight, torch.full_like(weight, 3)))

        recovered.train()
        # Both participating sets improve in epoch 4, while demo loss worsens.
        # Ties in epoch 5 must not replace that selected model.
        self.assertEqual(recovered.best_validation_loss_dict, {"first": (1.0, 3), "second": (2.0, 3)})
        self.assertEqual(recovered.best_model_epoch, 3)
        final = self.resume(recovered, epochs=5)
        self.assert_states_equal(final, recovered)
        self.assertEqual(final.best_model_epoch, 3)
        for weight in torch.load(Path(final.run_dir, "network.pth"), weights_only=True).values():
            self.assertTrue(torch.equal(weight, torch.full_like(weight, 4)))

    def test_legacy_checkpoint_is_upgraded_when_training_continues(self):
        original = self.make_session(epochs=1)
        original.train()
        self.make_legacy_checkpoint(original)
        recovered = self.resume(original)
        recovered.train()
        self.assertTrue(Path(recovered.run_dir, "states", SAVED_CHECKPOINT_NAME).is_file())
        self.assert_states_equal(self.resume(recovered), recovered)

    def test_best_checkpoint_weights_do_not_depend_on_inference_export(self):
        original = self.make_session(epochs=1, session_class=SelectionSession)
        original.train()
        original.config_session.num_epochs = 3
        Path(original.run_dir, "network.pth").unlink()
        original.train()
        recovered = self.resume(original)
        self.assertEqual(recovered.best_model_epoch, 0)
        for weight in torch.load(Path(original.run_dir, "network.pth"), weights_only=True).values():
            self.assertTrue(torch.equal(weight, torch.ones_like(weight)))

    def test_changed_validation_setup_is_rejected(self):
        original = self.make_session(epochs=1)
        original.train()
        checkpoint_path = Path(original.run_dir, "states", SAVED_CHECKPOINT_NAME)
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        for key, message in (("validation", "validation datasets"),
                             ("best_validation_loss_dict", "model-selection datasets")):
            with self.subTest(key=key):
                torch.save({**checkpoint, key: {}}, checkpoint_path)
                with self.assertRaisesRegex(ValueError, message):
                    self.resume(original)

    def test_checkpoint_without_selected_model_can_resume(self):
        original = self.make_session(epochs=1)
        method = original.do_one_validation_epoch

        def validate(name):
            method(name)
            original.value_logger_valid_dict[name].average_of_epoch["loss"] = float("inf")

        with patch.object(original, "do_one_validation_epoch", side_effect=validate):
            original.train()
        # A leftover temporary file from an abrupt process exit is not a checkpoint.
        torch.save({}, Path(original.run_dir, "states", ".checkpoint-leftover.tmp"))
        recovered = self.resume(original)
        self.assert_states_equal(recovered, original)
        self.assertIsNone(recovered.best_model_epoch)
        self.assertFalse(Path(original.run_dir, "network.pth").exists())


if __name__ == "__main__":
    unittest.main()
