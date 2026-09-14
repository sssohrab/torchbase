"""Exercise the runnable example, including real mid-epoch checkpoint recovery."""

import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from examples.image_reconstruction import MyTrainingSession, get_config, main
from torchbase.utils.session import RandomnessGeneratorStates


def load_tests(loader, tests, pattern):
    # Include the user-facing suite in the repository's normal unittest discovery.
    tests.addTests(loader.loadTestsFromName("examples.tests.test_image_reconstruction"))
    return tests


class ImageReconstructionExampleUnitTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(RandomnessGeneratorStates().apply)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        self.storage = tempfile.TemporaryDirectory()
        self.addCleanup(self.storage.cleanup)
        printing = patch.object(MyTrainingSession, "print")
        printing.start()
        self.addCleanup(printing.stop)

    def checkpoint(self, session):
        return torch.load(Path(session.run_dir, "states", "checkpoint.pth"), weights_only=True, map_location="cpu")

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_interrupted_training_and_each_validation_dataset_resume_exactly(self):
        config = get_config()
        config["session"]["num_epochs"] = 2
        config["data"].update(num_images=40, image_size=(8, 8), split_portions=(0.5, 0.5))
        initial_rng = RandomnessGeneratorStates()

        def session(tag):
            result = MyTrainingSession(config, runs_parent_dir=self.storage.name, tag_postfix=tag)
            self.addCleanup(result.writer.close)
            return result

        uninterrupted = session("uninterrupted")
        uninterrupted()
        expected = self.checkpoint(uninterrupted)
        expected_next_random = RandomnessGeneratorStates().to_dict()

        for phase in ("training", "train", "valid", "valid-aug"):
            with self.subTest(phase=phase):
                initial_rng.apply()
                interrupted = session("interrupted-" + phase)
                method = "do_one_training_iteration" if phase == "training" else "do_one_validation_iteration"
                original = getattr(interrupted, method)

                def stop_after_unsaved_batch(*args):
                    original(*args)
                    if phase == "training":
                        progress = interrupted.progress_train
                    elif args[1] == phase:
                        progress = interrupted.progress_valid_dict[phase]
                    else:
                        return
                    if progress.epoch == 0 and progress.iter_current_epoch == 3:
                        raise KeyboardInterrupt("Interrupt after batch 3; batch 2 was checkpointed.")

                with patch.object(interrupted, method, side_effect=stop_after_unsaved_batch):
                    with self.assertRaises(KeyboardInterrupt):
                        interrupted()
                saved = self.checkpoint(interrupted)
                progress = saved["progress_train"] if phase == "training" else saved["validation"][phase]["progress"]
                self.assertEqual(progress["iter_current_epoch"], 2)

                # Recovery must not depend on randomness consumed after the interruption.
                torch.rand(10)
                np.random.rand(10)
                random.random()
                recovered = main(["--resume", interrupted.run_dir])
                self.assertEqual(Path(recovered.run_dir).resolve(), Path(interrupted.run_dir).resolve())
                self.assert_nested_equal(self.checkpoint(recovered), expected)
                self.assert_nested_equal(RandomnessGeneratorStates().to_dict(), expected_next_random)
                self.assert_nested_equal(torch.load(Path(recovered.run_dir, "network.pth"), weights_only=True),
                                         expected["best_model"])

    def test_cli_start_extend_and_inference_snippet(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "examples.image_reconstruction", "--runs-dir", self.storage.name, "--epochs", "1"],
            cwd=root, env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        run_dir, = Path(self.storage.name).iterdir()
        self.assertIn("Run directory: {}".format(run_dir), result.stdout)
        config_path = run_dir / "config.json"
        original_config = config_path.read_bytes()
        self.assertEqual(json.loads(original_config)["session"]["num_epochs"], 1)

        extended = main(["--resume", str(run_dir), "--epochs", "2"])
        self.assertEqual(extended.progress_train.epoch, 2)
        self.assertEqual(config_path.read_bytes(), original_config)
        # A subsequent recovery uses the latest snapshot, not the original one-epoch config.
        completed = main(["--resume", str(run_dir)])
        self.assertEqual(completed.config_session.num_epochs, 2)
        self.assert_nested_equal(self.checkpoint(completed), self.checkpoint(extended))

        guide = (root / "examples" / "README.md").read_text()
        code, = re.findall(r"^```python\n(.*?)^```", guide, flags=re.MULTILINE | re.DOTALL)
        namespace = {}
        exec(compile(code.replace("runs/YOUR_RUN_TAG", run_dir.as_posix()), "examples/README.md", "exec"), namespace)
        probabilities = namespace["probabilities"]
        self.assertEqual(tuple(probabilities.shape), (1, 2, 32, 32))
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all().item())
        self.assertFalse(namespace["network"].training)

    def test_cli_rejects_invalid_requests_without_creating_a_run(self):
        for arguments in (["--epochs", "0", "--runs-dir", self.storage.name],
                          ["--resume", self.storage.name],
                          ["--resume", self.storage.name, "--runs-dir", self.storage.name]):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit) as error:
                main(arguments)
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(list(Path(self.storage.name).iterdir()), [])

    def test_configuration_is_fresh_for_each_new_experiment(self):
        edited = get_config()
        edited["data"]["num_images"] = 1
        edited["metrics"]["ImageReconstructionMetrics"].clear()
        fresh = get_config()
        self.assertEqual(fresh["data"]["num_images"], 20)
        self.assertEqual(fresh["metrics"]["ImageReconstructionMetrics"], ["psnr"])
