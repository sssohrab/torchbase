"""Test the experimental setup's contracts, not its final training accuracy."""

import math
import random
import tempfile
import unittest

import numpy as np
import torch

from examples.image_reconstruction import MyTrainingSession, get_config
from torchbase.utils.session import RandomnessGeneratorStates


class ImageReconstructionSetupTests(unittest.TestCase):
    def setUp(self):
        # Each test gets an independent, tiny CPU experiment. Restore global state
        # afterwards so these tests can run alongside other experiments' tests.
        self.addCleanup(RandomnessGeneratorStates().apply)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        storage = tempfile.TemporaryDirectory()
        self.addCleanup(storage.cleanup)

        config = get_config()
        config["session"].update(num_epochs=1, mini_batch_size=2)
        config["data"].update(num_images=10, image_size=(8, 8), split_portions=(0.6, 0.4))
        self.session = MyTrainingSession(config, runs_parent_dir=storage.name)
        self.addCleanup(self.session.writer.close)
        # Direct dataset access leaves the training dataloader's position untouched.
        self.batch = self.session.dataset_train[:2]

    def test_datasets_have_expected_shapes_types_and_binary_targets(self):
        self.assertEqual(len(self.session.dataset_train), 6)
        validation = self.session.datasets_valid_dict
        self.assertEqual([len(dataset) for dataset in validation.datasets], [6, 4, 4])
        for dataset in (self.session.dataset_train, *validation.datasets):
            with self.subTest(samples=len(dataset)):
                self.assertEqual(dataset.column_names, ["image"])
                images = dataset[:]["image"]
                self.assertEqual(tuple(images.shape), (len(dataset), 2, 8, 8))
                self.assertEqual(images.dtype, torch.float32)
                self.assertEqual(images.device.type, "cpu")
                self.assertTrue(((images == 0) | (images == 1)).all().item())

    def test_monitoring_dataset_does_not_participate_in_model_selection(self):
        validation = self.session.datasets_valid_dict
        self.assertEqual(validation.names, ("train", "valid", "valid-aug"))
        self.assertEqual(validation.only_for_demo, (True, False, False))
        self.assertIs(validation.datasets[0], self.session.dataset_train)
        self.assertEqual(set(self.session.best_validation_loss_dict), {"valid", "valid-aug"})

    def test_forward_pass_supplies_matching_loss_and_metric_inputs(self):
        outputs = self.session.forward_pass(self.batch)
        self.assertEqual(set(outputs), {"output", "target", "gt_for_metrics", "predictions_for_metrics"})
        for tensor in outputs.values():
            self.assertEqual(tuple(tensor.shape), (2, 2, 8, 8))
            self.assertTrue(torch.isfinite(tensor).all().item())
        self.assertTrue(torch.equal(outputs["target"], self.batch["image"]))
        self.assertTrue(torch.equal(outputs["gt_for_metrics"], outputs["target"]))
        torch.testing.assert_close(outputs["predictions_for_metrics"], outputs["output"].sigmoid())

    def test_loss_has_a_known_value_and_backpropagates_finite_gradients(self):
        # Zero logits mean probability 1/2: mean binary cross-entropy is log(2).
        known_loss = self.session.loss_function(output=torch.zeros_like(self.batch["image"]),
                                                target=self.batch["image"])
        self.assertAlmostEqual(known_loss.item(), math.log(2), places=6)

        outputs = self.session.forward_pass(self.batch)
        loss = self.session.loss_function(output=outputs["output"], target=outputs["target"])
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(loss.requires_grad)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for name, parameter in self.session.network.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
        # Some zero gradients are legitimate (e.g. inactive ReLUs).
        self.assertTrue(any(parameter.grad.abs().sum().item() > 0
                            for parameter in self.session.network.parameters()))

    def test_configured_metrics_match_handcrafted_predictions(self):
        truth = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])
        probabilities = torch.tensor([[[[0.1, 0.8], [0.7, 0.9]]]])
        # Use the session's logger so this also exercises the example's keyword maps.
        values = self.session.loggable_train(loss_tensor=torch.tensor(0.5),
                                            gt_for_metrics=truth, predictions_for_metrics=probabilities)
        self.assertEqual(set(values), {"loss", "precision_micro", "f1_score_micro", "psnr"})
        self.assertEqual(values["loss"], 0.5)
        # Three of four pixels are correct; binary micro precision/F1 equal accuracy.
        self.assertAlmostEqual(values["precision_micro"], 0.75)
        self.assertAlmostEqual(values["f1_score_micro"], 0.75)
        self.assertAlmostEqual(values["psnr"], 10 * math.log10(1 / (0.1875 + 1e-9)), places=5)

    def test_one_training_iteration_updates_weights_progress_and_logs(self):
        before = {name: value.clone() for name, value in self.session.network.state_dict().items()}
        self.session.do_one_training_iteration(self.batch)
        self.assertTrue(any(not torch.equal(value, before[name])
                            for name, value in self.session.network.state_dict().items()))
        self.assertEqual(self.session.progress_train.iter_total, 1)
        self.assertEqual(self.session.progress_train.samples_total, 2)
        values = self.session.value_logger_train.current_values
        self.assertEqual(set(values), {"loss", "precision_micro", "f1_score_micro", "psnr"})
        self.assertTrue(all(math.isfinite(value) for value in values.values()))
        for progress in self.session.progress_valid_dict.values():
            self.assertEqual(progress.iter_total, 0)


if __name__ == "__main__":
    unittest.main()
