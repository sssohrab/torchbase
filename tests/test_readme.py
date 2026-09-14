"""Run the README's actual example rather than maintaining a separate copy."""

import math
import os
from pathlib import Path
import re
import tempfile
import unittest

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from torchbase.utils.session import RandomnessGeneratorStates


class ReadmeExampleUnitTest(unittest.TestCase):
    def test_image_reconstruction_example_trains_and_recovers(self):
        readme = Path(__file__).resolve().parents[1] / "README.md"
        blocks = re.findall(r"^```python\n(.*?)^```", readme.read_text(), flags=re.MULTILINE | re.DOTALL)
        namespace = {"__name__": __name__}

        def run_block(marker):
            matches = [block for block in blocks if marker in block]
            self.assertEqual(len(matches), 1, "Expected one README block containing {!r}".format(marker))
            exec(compile(matches[0], str(readme), "exec"), namespace)

        self.addCleanup(RandomnessGeneratorStates().apply)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        storage = tempfile.TemporaryDirectory()
        self.addCleanup(storage.cleanup)
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(storage.name)

        run_block("def init_datasets(")
        run_block("config = {")
        run_block("session = MyTrainingSession(config)")
        session = namespace["session"]
        self.addCleanup(session.writer.close)
        config = namespace["config"]
        run_block("session()\n")

        self.assertEqual(session.progress_train.epoch, config["session"]["num_epochs"])
        self.assertEqual(session.progress_train.samples_total,
                         len(session.dataset_train) * config["session"]["num_epochs"])
        for name, dataset in zip(session.datasets_valid_dict.names, session.datasets_valid_dict.datasets):
            self.assertEqual(session.progress_valid_dict[name].samples_total,
                             len(dataset) * config["session"]["num_epochs"])
        batch = next(iter(session.dataloader_train))
        outputs = session.forward_pass(batch)
        expected_shape = (len(batch["image"]), config["network"]["num_ch"], *config["data"]["image_size"])
        for tensor in outputs.values():
            self.assertEqual(tuple(tensor.shape), expected_shape)
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertTrue(torch.isfinite(tensor).all().item())
        self.assertTrue(((outputs["target"] == 0) | (outputs["target"] == 1)).all().item())
        self.assertTrue(torch.equal(outputs["gt_for_metrics"], outputs["target"]))
        self.assertTrue(torch.equal(outputs["predictions_for_metrics"], outputs["output"].sigmoid()))
        loss = session.loss_function(output=outputs["output"], target=outputs["target"])
        self.assertTrue(loss.requires_grad)
        self.assertTrue(torch.isfinite(loss).item())
        for logger in (session.value_logger_train, *session.value_logger_valid_dict.values()):
            self.assertEqual(set(logger.average_of_epoch), {"loss", "precision_micro", "f1_score_micro", "psnr"})
            self.assertTrue(all(math.isfinite(value) for value in logger.average_of_epoch.values()))

        run_dir = Path(session.run_dir)
        self.assertTrue((run_dir / "states" / "checkpoint.pth").is_file())
        self.assertTrue((run_dir / "network.pth").is_file())
        events = EventAccumulator(str(run_dir)).Reload()
        for prefix in ("training", *("validation-" + name for name in session.datasets_valid_dict.names)):
            self.assertIn(prefix + "/loss/epochs", events.Tags()["scalars"])
            self.assertIn(prefix + "/psnr/batch_means", events.Tags()["scalars"])
            self.assertIn(prefix + "/f1_score_micro/epochs", events.Tags()["scalars"])

        namespace.update(same_config_as_before=config, the_tag_to_the_suspended_experiment=run_dir.name)
        run_block("the_recovered_session = MyTrainingSession(")
        recovered = namespace["the_recovered_session"]
        self.addCleanup(recovered.writer.close)
        self.assertEqual(recovered.run_dir, session.run_dir)
        self.assertEqual(recovered.progress_train.state_dict(), session.progress_train.state_dict())
        self.assertEqual(recovered.value_logger_train.state_dict(), session.value_logger_train.state_dict())
        for name in session.datasets_valid_dict.names:
            self.assertEqual(recovered.value_logger_valid_dict[name].state_dict(),
                             session.value_logger_valid_dict[name].state_dict())
        for name, value in session.network.state_dict().items():
            self.assertTrue(torch.equal(recovered.network.state_dict()[name], value))
