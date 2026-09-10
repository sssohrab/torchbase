"""Small CPU experiments exercising real save/load/continue behavior."""

from dataclasses import asdict
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset
import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from torchbase import TrainingBaseSession
from torchbase.session import SAVED_RNG_NAME, SAVED_CHECKPOINT_NAME
from torchbase.utils.checkpoint import atomic_torch_save
from torchbase.utils.data import ValidationDatasetsDict
from torchbase.utils.session import RandomnessGeneratorStates


class ResumeSession(TrainingBaseSession):
    def __init__(self, *args, **kwargs):
        self.seen_batches = []
        super().__init__(*args, **kwargs)

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
        size = self.config_data.get("size", 8)
        inputs = torch.rand(size, 2) + np.random.random() + random.random()
        dataset = Dataset.from_dict({"inputs": inputs, "target": inputs.sum(1, keepdim=True)})
        dataset.set_format("torch")
        return dataset, ValidationDatasetsDict(
            datasets=(dataset.select(range(min(size, self.config_data.get("valid_size", 5)))),
                      dataset.select(range(min(size, self.config_data.get("demo_size", 3))))),
            only_for_demo=(False, True), names=("valid", "demo"),
        )

    def init_network(self):
        return torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Dropout(0.25))

    def init_metrics(self):
        return None

    def forward_pass(self, mini_batch):
        self.seen_batches.append((self._phase, self._validation_index, mini_batch["inputs"].tolist()))
        noise = random.random() * 0.01 + np.random.random() * 0.01
        return {"output": self.network(mini_batch["inputs"] + noise), "target": mini_batch["target"]}

    def loss_function(self, *, output, target):
        return torch.nn.functional.mse_loss(output, target)

    @staticmethod
    def print(report, end=None):
        pass


class AugmentedSession(ResumeSession):
    @staticmethod
    def dataloader_collate_function(batch):
        batch = torch.utils.data.default_collate(batch)
        batch["inputs"] += torch.rand_like(batch["inputs"]) * 0.01 + np.random.random() * 0.01 + random.random() * 0.01
        return batch


class WorkerSession(ResumeSession):
    def init_dataloaders(self):
        train, validation = super().init_dataloaders()
        # Use workers for the interrupted loader; keep unrelated loaders cheap.
        # This exercises prefetch/recovery without repeatedly spawning idle workers.
        def single_process(loader):
            return StatefulDataLoader(loader.dataset, batch_size=loader.batch_size, shuffle=True,
                                      generator=loader.generator, collate_fn=loader.collate_fn)

        phase = self.config_data["worker_phase"]
        if phase != "training":
            train = single_process(train)
        return train, {name: loader if name == phase else single_process(loader)
                       for name, loader in validation.items()}


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

    def make_session(self, *, epochs=3, source=None, fresh=True, tag="run", session_class=ResumeSession,
                     interval=100, workers=0, data=None, batch_size=3):
        config = ResumeSession.get_config(epochs)
        config["session"].update(checkpoint_interval=interval, dataloader_num_workers=workers,
                                  mini_batch_size=batch_size)
        config["data"] = data or {}
        session = session_class(
            config=config, runs_parent_dir=self.storage.name,
            source_run_dir_tag=source, create_run_dir_afresh=fresh,
            tag_postfix=tag if fresh else None,
        )
        self.addCleanup(session.writer.close)
        self.addCleanup(self.close_loaders, session)
        return session

    @staticmethod
    def close_loaders(session):
        # A simulated interruption keeps the process alive, unlike a real crash.
        # Explicitly close test workers rather than leaving them for GC.
        for loader in [session.dataloader_train, *session.dataloader_valid_dict.values()]:
            iterator = getattr(loader, "_iterator", None)
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()

    def resume(self, session, *, epochs=3):
        return self.make_session(epochs=epochs, source=Path(session.run_dir).name, fresh=False,
                                 session_class=type(session), interval=session.config_session.checkpoint_interval,
                                 workers=session.config_session.dataloader_num_workers, data=session.config_data,
                                 batch_size=session.config_session.mini_batch_size)

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
        self.assert_nested_equal(actual.state_dict(), expected.state_dict())

    def assert_states_equal(self, actual, expected):
        self.assertEqual(actual.best_validation_loss_dict, expected.best_validation_loss_dict)
        self.assertEqual(actual.best_model_epoch, expected.best_model_epoch)
        self.assertEqual(actual._phase, expected._phase)
        self.assertEqual(actual._validation_index, expected._validation_index)
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

    def test_saved_config_can_construct_and_resume_a_session_without_mutating_input(self):
        config = ResumeSession.get_config(num_epochs=1)
        config["session"]["loss_function_params"] = {"options": {"weights": [1.0, 2.0]}}
        config["data"]["custom"] = {"shape": [2, 1]}
        config["metrics"]["custom"] = ["score"]
        config["network"]["custom"] = {"layers": [2, 1]}
        original = deepcopy(config)
        session = ResumeSession(config, runs_parent_dir=self.storage.name, tag_postfix="config-original")
        self.addCleanup(session.writer.close)
        self.assertEqual(config, original)
        saved = json.loads(Path(session.run_dir, "config.json").read_text())
        self.assertEqual(saved["session"], session.config_session.to_dict())
        for section in ("data", "metrics", "network"):
            self.assertEqual(saved[section], original[section])
        session.train()

        fresh = ResumeSession(saved, runs_parent_dir=self.storage.name, tag_postfix="config-fresh")
        self.addCleanup(fresh.writer.close)
        resumed = ResumeSession(saved, runs_parent_dir=self.storage.name, create_run_dir_afresh=False,
                                source_run_dir_tag=Path(session.run_dir).name)
        self.addCleanup(resumed.writer.close)
        self.assert_states_equal(resumed, session)
        self.assertEqual(json.loads(Path(fresh.run_dir, "config.json").read_text()), saved)
        recovered_configs = list(Path(session.run_dir).glob("config_*.json"))
        self.assertEqual(len(recovered_configs), 1)
        self.assertEqual(json.loads(recovered_configs[0].read_text()), saved)
        for current in (session, fresh, resumed):
            current.config_session.loss_function_params["options"]["weights"].clear()
            current.config_data["custom"]["shape"].clear()
            current.config_metrics["custom"].clear()
            current.config_network["custom"]["layers"].clear()
        self.assertEqual(config, original)
        self.assertEqual(saved["metrics"], original["metrics"])
        self.assertEqual(saved["data"], original["data"])
        self.assertEqual(saved["network"], original["network"])
        self.assertEqual(saved["session"]["loss_function_params"], original["session"]["loss_function_params"])

    def test_invalid_validation_sets_fail_before_dataloader_initialization(self):
        dataset = Dataset.from_dict({"inputs": [[1.0, 2.0]], "target": [[3.0]]})
        invalid_sets = (
            ValidationDatasetsDict((dataset,), (False,), ()),
            ValidationDatasetsDict((dataset, dataset), (False, True), ("valid", "valid")),
            ValidationDatasetsDict((), (), ()),
            ValidationDatasetsDict((dataset.select([]),), (False,), ("valid",)),
        )
        for index, validation in enumerate(invalid_sets):
            with self.subTest(index=index), \
                    patch.object(ResumeSession, "init_datasets", return_value=(dataset, validation)), \
                    patch.object(ResumeSession, "init_dataloaders") as init_loaders:
                with self.assertRaisesRegex(ValueError, "valid `datasets_valid_dict`"):
                    self.make_session(tag="invalid-validation-{}".format(index))
                init_loaders.assert_not_called()

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

    def test_missing_checkpoint_is_not_silently_reset(self):
        original = self.make_session(epochs=1)
        original.train()
        # Only this test's disposable checkpoint is modified.
        Path(original.run_dir, "states", SAVED_CHECKPOINT_NAME).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "v0.1.x"):
            self.resume(original)

    def test_interrupted_training_or_validation_replays_from_phase_boundary(self):
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
                method_name = ("do_one_training_iteration" if phase == "training"
                               else "do_one_validation_iteration")
                method = getattr(resumed, method_name)

                def interrupt(*args):
                    method(*args)
                    if phase != "training" and args[1] != phase:
                        return
                    raise KeyboardInterrupt("simulated interruption")

                with patch.object(resumed, method_name, side_effect=interrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        resumed()
                checkpoint = torch.load(checkpoint_path, weights_only=True)
                recovered = self.resume(original)
                self.assertEqual(recovered.progress_train.epoch, 1 if phase == "training" else 2)
                self.assertEqual(checkpoint["phase"], "training" if phase == "training" else "validation")
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

    def exercise_mid_epoch_recovery(self, *, phase="training", workers=0, epoch=0,
                                    fail_at=5, interval=2, after_save=False, session_class=ResumeSession,
                                    interrupt_again=False):
        data = {"size": 17, "valid_size": 17, "demo_size": 17}
        if workers:
            data["worker_phase"] = phase
        tag = "{}-{}-{}-{}-{}-{}".format(phase, workers, epoch, fail_at, interval, after_save)
        self.seed()
        epochs = 2 if workers else 3
        expected = self.make_session(tag="expected-" + tag, data=data, workers=workers, interval=interval,
                                     epochs=epochs, session_class=session_class)
        expected.train()
        expected_rng = RandomnessGeneratorStates().to_dict()

        self.seed()
        original = self.make_session(tag="interrupted-" + tag, data=data, workers=workers, interval=interval,
                                     epochs=epochs, session_class=session_class)
        progress = original.progress_train if phase == "training" else original.progress_valid_dict[phase]
        active_phase = "training" if phase == "training" else "validation"
        active_index = 0 if phase in ("training", "valid") else 1
        method_name = ("_save_checkpoint" if after_save else
                       "do_one_training_iteration" if phase == "training" else "do_one_validation_iteration")
        method = getattr(original, method_name)

        def interrupt(*args, **kwargs):
            method(*args, **kwargs)
            if (original._phase == active_phase and original._validation_index == active_index
                    and progress.epoch == epoch and progress.iter_current_epoch == fail_at):
                raise KeyboardInterrupt("simulated crash")

        with patch.object(original, method_name, side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                original()
        self.close_loaders(original)
        checkpoint = torch.load(Path(original.run_dir, "states", SAVED_CHECKPOINT_NAME), weights_only=True)
        saved_progress = (checkpoint["progress_train"] if phase == "training"
                          else checkpoint["validation"][phase]["progress"])
        saved_iteration = fail_at if after_save else ((fail_at - 1) // interval) * interval
        self.assertEqual(saved_progress["iter_current_epoch"], saved_iteration)
        self.assertEqual(checkpoint["phase"], active_phase)
        saved_batches = checkpoint["progress_train"]["iter_total"] + sum(
            state["progress"]["iter_total"] for state in checkpoint["validation"].values())

        # External work between processes must not affect continuation.
        torch.rand(20)
        np.random.rand(20)
        random.random()
        recovered = self.resume(original, epochs=epochs)
        if interrupt_again:
            validate = recovered.do_one_validation_iteration

            def second_interruption(batch, name):
                validate(batch, name)
                if name == "valid" and recovered.progress_valid_dict[name].iter_current_epoch == 5:
                    raise KeyboardInterrupt()

            with patch.object(recovered, "do_one_validation_iteration", side_effect=second_interruption):
                with self.assertRaises(KeyboardInterrupt):
                    recovered()
            checkpoint = torch.load(Path(original.run_dir, "states", SAVED_CHECKPOINT_NAME), weights_only=True)
            saved_batches = checkpoint["progress_train"]["iter_total"] + sum(
                state["progress"]["iter_total"] for state in checkpoint["validation"].values())
            recovered = self.resume(original, epochs=epochs)
        recovered.train()
        self.assertEqual(recovered.seen_batches, expected.seen_batches[saved_batches:])
        self.assert_states_equal(recovered, expected)
        self.assertEqual(RandomnessGeneratorStates().to_dict(), expected_rng)
        self.assert_nested_equal(
            torch.load(Path(recovered.run_dir, "network.pth"), weights_only=True),
            torch.load(Path(expected.run_dir, "network.pth"), weights_only=True))

    def test_mid_epoch_training_and_each_validation_dataset_resume_exactly(self):
        for phase in ("training", "valid", "demo"):
            with self.subTest(phase=phase):
                self.exercise_mid_epoch_recovery(phase=phase)

    def test_later_epoch_resumes_without_resetting_epoch_averages(self):
        self.exercise_mid_epoch_recovery(epoch=1)

    def test_checkpoint_after_last_uneven_batch_resumes_without_repeating_epoch(self):
        for phase in ("training", "valid", "demo"):
            with self.subTest(phase=phase):
                self.exercise_mid_epoch_recovery(phase=phase, fail_at=6, after_save=True)

    def test_interval_one_and_exact_checkpoint_boundary(self):
        self.exercise_mid_epoch_recovery(fail_at=3, interval=1, after_save=True)

    def test_prefetched_worker_batches_are_not_skipped_on_recovery(self):
        self.exercise_mid_epoch_recovery(workers=2, session_class=WorkerSession)

    def test_prefetched_validation_batches_resume_exactly(self):
        self.exercise_mid_epoch_recovery(phase="demo", workers=2, session_class=WorkerSession)

    def test_prefetched_last_batch_checkpoint_resumes_exactly(self):
        self.exercise_mid_epoch_recovery(workers=2, session_class=WorkerSession, fail_at=6, after_save=True)

    def test_repeated_interruptions_in_training_then_validation(self):
        self.exercise_mid_epoch_recovery(interrupt_again=True)

    def test_iteration_9070_recovers_iteration_9000_of_a_long_epoch(self):
        original = self.make_session(epochs=1, data={"size": 10000}, batch_size=1, interval=100)
        iteration = original.do_one_training_iteration

        def interrupt(batch):
            iteration(batch)
            if original.progress_train.iter_current_epoch == 9070:
                raise KeyboardInterrupt()

        with patch.object(original, "do_one_training_iteration", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                original()
        recovered = self.resume(original, epochs=1)
        self.assertEqual(recovered.progress_train.iter_current_epoch, 9000)
        self.assertEqual(recovered.progress_train.samples_current_epoch, 9000)
        recovered.train()
        replayed_training = [batch for batch in recovered.seen_batches if batch[0] == "training"]
        self.assertEqual(len(replayed_training), 1000)
        self.assertEqual(replayed_training[:70], original.seen_batches[9000:9070])
        self.assertEqual(recovered.progress_train.iter_total, 10000)
        self.assertEqual(recovered.progress_train.epoch, 1)

    def test_random_collation_with_zero_workers_resumes_exactly(self):
        self.exercise_mid_epoch_recovery(session_class=AugmentedSession)

    def test_unsupported_loader_modes_are_rejected(self):
        original = self.make_session()
        dataset = original.dataset_train
        for options in ({}, {"generator": torch.Generator(), "in_order": False},
                        {"generator": torch.Generator(), "num_workers": 1, "persistent_workers": True}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "Recovery"):
                original._dataloader_settings(StatefulDataLoader(dataset, **options))
        with self.assertRaisesRegex(ValueError, "StatefulDataLoader"):
            original._dataloader_settings(torch.utils.data.DataLoader(dataset))

    def test_phase_boundaries_resume_without_repeating_completed_work(self):
        for phase, index in (("validation", 0), ("validation", 1), ("selection", 2)):
            with self.subTest(phase=phase, index=index):
                self.seed()
                expected = self.make_session(tag="expected-{}-{}".format(phase, index))
                expected.train()
                expected_rng = RandomnessGeneratorStates().to_dict()
                self.seed()
                original = self.make_session(tag="interrupted-{}-{}".format(phase, index))
                save = original._save_checkpoint

                def interrupt(**kwargs):
                    save(**kwargs)
                    if original._phase == phase and original._validation_index == index:
                        raise KeyboardInterrupt()

                with patch.object(original, "_save_checkpoint", side_effect=interrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        original()
                completed = len(original.seen_batches)
                recovered = self.resume(original)
                recovered.train()
                self.assertEqual(recovered.seen_batches, expected.seen_batches[completed:])
                self.assert_states_equal(recovered, expected)
                self.assertEqual(RandomnessGeneratorStates().to_dict(), expected_rng)

    def test_failed_periodic_save_preserves_previous_iteration(self):
        self.seed()
        expected = self.make_session(tag="expected", data={"size": 17}, interval=2)
        expected.train()
        expected_rng = RandomnessGeneratorStates().to_dict()
        self.seed()
        original = self.make_session(tag="interrupted", data={"size": 17}, interval=2)
        save = original._save_checkpoint

        def fail_periodic_save(**kwargs):
            if original._phase == "training" and original.progress_train.iter_current_epoch == 4:
                with patch("torchbase.utils.checkpoint.os.replace", side_effect=OSError("disk failure")):
                    save(**kwargs)
            else:
                save(**kwargs)

        with patch.object(original, "_save_checkpoint", side_effect=fail_periodic_save):
            with self.assertRaisesRegex(OSError, "disk failure"):
                original()
        recovered = self.resume(original)
        self.assertEqual(recovered.progress_train.iter_current_epoch, 2)
        recovered.train()
        self.assert_states_equal(recovered, expected)
        self.assertEqual(RandomnessGeneratorStates().to_dict(), expected_rng)

    def test_changed_loader_configuration_is_rejected(self):
        original = self.make_session(epochs=1)
        original.train()
        for change in ({"batch_size": 4}, {"workers": 1}, {"data": {"size": 9}}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "dataloader settings"):
                self.make_session(source=Path(original.run_dir).name, fresh=False, **change)

    def test_missing_loader_position_is_rejected(self):
        original = self.make_session(interval=1)
        iteration = original.do_one_training_iteration

        def interrupt(batch):
            iteration(batch)
            if original.progress_train.iter_current_epoch == 2:
                raise KeyboardInterrupt()

        with patch.object(original, "do_one_training_iteration", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                original()
        path = Path(original.run_dir, "states", SAVED_CHECKPOINT_NAME)
        checkpoint = torch.load(path, weights_only=True)
        checkpoint["dataloader_train"]["iterator"] = None
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "dataloader position"):
            self.resume(original)

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

    def test_invalid_or_unsupported_checkpoint_is_rejected(self):
        original = self.make_session(epochs=1)
        original.train()
        checkpoint_path = Path(original.run_dir, "states", SAVED_CHECKPOINT_NAME)
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        for corrupt in ({}, {**checkpoint, "format_version": 1}, {**checkpoint, "format_version": 999}):
            with self.subTest(version=corrupt.get("format_version")):
                torch.save(corrupt, checkpoint_path)
                with self.assertRaisesRegex(ValueError, "checkpoint"):
                    self.resume(original)

    def test_inconsistent_loop_state_cannot_replace_checkpoint(self):
        original = self.make_session(epochs=1)
        original.train()
        checkpoint_path = Path(original.run_dir, "states", SAVED_CHECKPOINT_NAME)
        before = checkpoint_path.read_bytes()
        original._phase = "validation"
        with self.assertRaisesRegex(ValueError, "Inconsistent checkpoint"):
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
