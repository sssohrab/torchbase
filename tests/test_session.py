from torchbase.session import TrainingBaseSession
from torchbase.session import SAVED_RNG_NAME
from torchbase.utils.data import ValidationDatasetsDict, split_iterables

from torchbase.utils.metrics import BaseMetricsClass
from torchbase.utils.metrics_instances import BinaryClassificationMetrics, ImageReconstructionMetrics

import torch
from torchvision import transforms
from datasets import Dataset
import datasets

import unittest

from typing import Dict, Tuple, Any, List
import os
import random
import shutil
import time
import json
import inspect

TEST_STORAGE_DIR = os.path.join(os.path.split(os.path.abspath(__file__))[0], "storage")
os.makedirs(TEST_STORAGE_DIR, exist_ok=True)


class ExampleTrainingSessionClassStatic(TrainingBaseSession):
    @staticmethod
    def get_config() -> Dict:
        config = {
            "session": {
                "device_name": "cpu",
                "num_epochs": 1,
                "mini_batch_size": 4,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "dataloader_num_workers": 0,
            },
            "data": {
                "raw": {
                    "inputs": list(range(20)),
                    "labels": [i % 2 for i in range(20)],
                },
                "split_portions": (0.8, 0.2)
            },
            "metrics": {},
            "network": {
                "architecture": "SomeExampleNet",
                "num_ch": 2
            }
        }

        return config

    def init_datasets(self) -> Tuple[Dataset, ValidationDatasetsDict]:
        data_train, data_valid = split_iterables(self.config_data["raw"],
                                                 portions=self.config_data["split_portions"],
                                                 shuffle=True)

        def augment(item: Dict) -> Dict:
            item["inputs"] += random.randint(-5, +5)

            return item

        dataset_train_with_augmentation = Dataset.from_dict(data_train).map(lambda x: augment(x))
        dataset_train_without_augmentation = Dataset.from_dict(data_train)
        dataset_valid_with_augmentation = Dataset.from_dict(data_valid).map(lambda x: augment(x))
        dataset_valid_without_augmentation = Dataset.from_dict(data_valid)

        return (dataset_train_with_augmentation,
                ValidationDatasetsDict(
                    datasets=(dataset_train_without_augmentation,
                              dataset_valid_with_augmentation,
                              dataset_valid_without_augmentation),
                    only_for_demo=(True, False, False),
                    names=("train-no-aug", "valid-with-aug", "valid-no-aug")
                ))

    def init_network(self) -> torch.nn.Module:
        class SomeExampleNet(torch.nn.Module):
            def __init__(self, num_ch: int):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, num_ch, 3, padding=1)
                self.conv2 = torch.nn.Conv2d(num_ch, 1, 3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.conv1(x)
                x = torch.nn.ReLU()(x)
                x = self.conv2(x)

                return x

        network = SomeExampleNet(num_ch=self.config_network["num_ch"])

        return network

    def forward_pass(self, mini_batch: Dict[str, Any | torch.Tensor]) -> Dict[str, Any | torch.Tensor]:
        pass

    def loss_function(self, **kwargs: Any) -> torch.Tensor:
        pass

    def init_metrics(self) -> List[BaseMetricsClass] | None:
        pass


class TrainingBaseSessionStaticUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.path.exists(os.path.join(TEST_STORAGE_DIR)):
            shutil.rmtree(os.path.join(TEST_STORAGE_DIR))
        os.makedirs(TEST_STORAGE_DIR, exist_ok=True)
        # TODO: Create a temp folder in memory not disk

        cls.session_fresh_run_fresh_network = ExampleTrainingSessionClassStatic(
            config=ExampleTrainingSessionClassStatic.get_config(),
            runs_parent_dir=TEST_STORAGE_DIR,
            create_run_dir_afresh=True,
            source_run_dir_tag=None,
            tag_postfix="fresh-run-fresh-net"
        )

        cls.mock_train_steps_and_save_network_and_optimizer()

        time.sleep(1)  # To avoid creating the same tag again.

        cls.session_existing_run = ExampleTrainingSessionClassStatic(
            config=ExampleTrainingSessionClassStatic.get_config(),
            runs_parent_dir=TEST_STORAGE_DIR,
            create_run_dir_afresh=False,
            source_run_dir_tag=os.path.split(cls.session_fresh_run_fresh_network.run_dir)[-1]
        )

        cls.session_fresh_run_pretrained_network = ExampleTrainingSessionClassStatic(
            config=ExampleTrainingSessionClassStatic.get_config(),
            runs_parent_dir=TEST_STORAGE_DIR,
            create_run_dir_afresh=True,
            source_run_dir_tag=os.path.split(cls.session_fresh_run_fresh_network.run_dir)[-1],
            tag_postfix="fresh-run-existing-net"
        )

    @classmethod
    def tearDown(cls) -> None:
        cls.session_fresh_run_fresh_network.writer.close()
        cls.session_existing_run.writer.close()
        cls.session_fresh_run_pretrained_network.writer.close()

    @classmethod
    def mock_train_steps_and_save_network_and_optimizer(cls):
        for _ in range(5):
            cls.session_fresh_run_fresh_network.optimizer.zero_grad()
            tensor_inp = torch.randn(4, 1, 32, 32).to(cls.session_fresh_run_fresh_network.device)
            tensor_out = cls.session_fresh_run_fresh_network.network(tensor_inp)
            loss = torch.nn.MSELoss()(tensor_inp, tensor_out)
            loss.backward()
            cls.session_fresh_run_fresh_network.optimizer.step()

        cls.session_fresh_run_fresh_network.save_network_and_optimizer_states()
        for valid_dataset_name in cls.session_fresh_run_fresh_network.datasets_valid_dict.names:
            cls.session_fresh_run_fresh_network.save_progress_and_log_states_for_valid_set(valid_dataset_name)

    def test_instantiate_session_with_fresh_run_fresh_network(self):
        self.assertIsNotNone(self.session_fresh_run_fresh_network)
        self.assertIsInstance(self.session_fresh_run_fresh_network, TrainingBaseSession)

    def test_instantiate_session_with_existing_run(self):
        self.assertIsNotNone(self.session_existing_run)
        self.assertIsInstance(self.session_existing_run, TrainingBaseSession)

    def test_instantiate_session_with_fresh_run_pretrained_network(self):
        self.assertIsNotNone(self.session_fresh_run_pretrained_network)
        self.assertIsInstance(self.session_fresh_run_pretrained_network, TrainingBaseSession)

    def test_datasets_random_access_as_torch_tensor(self):
        all_datasets = ([self.session_fresh_run_fresh_network.dataset_train]
                        + list(self.session_fresh_run_fresh_network.datasets_valid_dict.datasets))
        for dataset in all_datasets:
            idx = random.randint(0, dataset.__len__() - 1)
            for expected_keys in ["inputs", "labels"]:
                self.assertIn(expected_keys, dataset[idx])
                self.assertIsInstance(dataset[idx][expected_keys], torch.Tensor)

    def test_saved_random_states_replicability(self):
        dataset_initial = self.session_fresh_run_fresh_network.dataset_train
        dataset_replicated = self.session_existing_run.dataset_train
        self.assertEqual(dataset_initial.data, dataset_replicated.data)

    def test_existing_run_dir_but_no_run_tag_specified(self):
        with self.assertRaises(ValueError):
            ExampleTrainingSessionClassStatic(config=ExampleTrainingSessionClassStatic.get_config(),
                                              runs_parent_dir=TEST_STORAGE_DIR,
                                              create_run_dir_afresh=False,
                                              source_run_dir_tag="Some-non-existing-tag")

    def test_config_saved_in_run_dir(self):
        run_dir = self.session_fresh_run_fresh_network.run_dir
        run_dir_content = os.listdir(run_dir)
        self.assertIn("config.json", run_dir_content)
        with open(os.path.join(run_dir, "config.json"), "r") as file:
            saved_config = json.load(file)

        self.assertEqual(saved_config["session"], self.session_fresh_run_fresh_network.config_session.to_dict())
        self.assertEqual(saved_config["data"]["raw"], self.session_fresh_run_fresh_network.config_data["raw"])
        self.assertEqual(saved_config["network"], self.session_fresh_run_fresh_network.config_network)
        self.assertEqual(saved_config["metrics"], self.session_fresh_run_fresh_network.config_metrics)

    def test_network_init_and_declared_architecture_mismatch(self):
        wrong_config = ExampleTrainingSessionClassStatic.get_config()
        wrong_config["network"]["architecture"] = "SomeMistakenlyHeldNetworkName"
        with self.assertRaises(TypeError):
            time.sleep(1)  # To avoid creating the same tag again.
            ExampleTrainingSessionClassStatic(config=wrong_config, runs_parent_dir=TEST_STORAGE_DIR)

    def test_network_loading(self):
        network_saved = self.session_fresh_run_fresh_network.network
        network_recovered = self.session_existing_run.network
        network_pretrained = self.session_fresh_run_pretrained_network.network
        self.assertTrue(torch.allclose(network_recovered.conv1.weight, network_saved.conv1.weight))
        self.assertTrue(torch.allclose(network_recovered.conv2.weight, network_saved.conv2.weight))
        self.assertTrue(torch.allclose(network_pretrained.conv1.weight, network_saved.conv1.weight))
        self.assertTrue(torch.allclose(network_pretrained.conv2.weight, network_saved.conv2.weight))

    def test_optimizer_loading(self):
        optimizer_saved = self.session_fresh_run_fresh_network.optimizer
        optimizer_recovered = self.session_existing_run.optimizer
        optimizer_reset = self.session_fresh_run_pretrained_network.optimizer

        self.assertTrue(
            torch.allclose(
                optimizer_recovered.state_dict()["state"][0]["exp_avg"],
                optimizer_saved.state_dict()["state"][0]["exp_avg"]
            )
        )

        self.assertTrue(0 not in optimizer_reset.state_dict()["state"])

    def test_invalid_source_states_dir(self):
        source_session = self.session_fresh_run_pretrained_network
        os.remove(os.path.join(source_session.run_dir, "states", SAVED_RNG_NAME))
        time.sleep(1)  # To avoid creating the same tag again.
        with self.assertRaises(FileNotFoundError):
            ExampleTrainingSessionClassStatic(
                config=ExampleTrainingSessionClassStatic.get_config(),
                runs_parent_dir=TEST_STORAGE_DIR,
                create_run_dir_afresh=False,
                source_run_dir_tag=os.path.split(source_session.run_dir)[-1]
            )

    def test_dataloader_instantiation(self):
        self.assertIsInstance(self.session_fresh_run_fresh_network.dataloader_train, torch.utils.data.DataLoader)
        self.assertIsInstance(self.session_fresh_run_fresh_network.dataloader_valid_dict, Dict)
        for dataset_name in self.session_fresh_run_fresh_network.datasets_valid_dict.names:
            self.assertIsInstance(self.session_fresh_run_fresh_network.dataloader_valid_dict[dataset_name],
                                  torch.utils.data.DataLoader)

    def test_dataloader_basic_functionality(self):

        dataloader = self.session_existing_run.dataloader_train
        mini_batch = next(iter(dataloader))
        self.assertEqual(mini_batch["inputs"].shape[0], self.session_existing_run.config_session.mini_batch_size)
        self.assertEqual(mini_batch["labels"].shape[0], self.session_existing_run.config_session.mini_batch_size)


class ExampleTrainingSessionClassDynamic(TrainingBaseSession):
    @staticmethod
    def get_config() -> Dict:
        config = {
            "session": {
                "device_name": "cpu",
                "num_epochs": 10,
                "mini_batch_size": 6,
                "learning_rate": 0.01,
                "weight_decay": 1e-6,
                "dataloader_num_workers": 0,
            },
            "data": {
                "num_samples": 20,
                "image_size": (32, 32),
                "split_portions": (0.8, 0.2)
            },
            "metrics": {
                "BinaryClassificationMetrics": [
                    "precision_micro", "recall_micro", "f1_score_micro"
                ],
                "ImageReconstructionMetrics": [
                    "psnr"
                ]
            },
            "network": {
                "architecture": "SomeSimpleCNN",
                "num_ch": 2
            }
        }

        return config

    def init_datasets(self) -> Tuple[Dataset, ValidationDatasetsDict]:
        def generate_random_data_for_test(num_samples: int, image_size: Tuple[int, int]) -> Dict[str, List]:
            data = {"image": [], "image_noisy": []}

            for _ in range(num_samples):
                image = torch.zeros(3, *image_size)
                num_rectangles = random.randint(0, 3)
                for _ in range(num_rectangles):
                    x1, y1 = torch.randint(0, image_size[0] // 2, (2,))
                    x2, y2 = torch.randint(image_size[0] // 2, image_size[0], (2,))
                    color = torch.rand(3)

                    image[:, x1:x2, y1:y2] = color.unsqueeze(1).unsqueeze(2)

                noise = torch.randn_like(image) * 0.2
                image_noisy = torch.clamp(image + noise, 0, 1)

                data["image"].append(image)
                data["image_noisy"].append(image_noisy)

            return data

        def augment(item: Dict) -> Dict:
            rotation = transforms.RandomRotation(degrees=(-30, 30))  # Rotate between -30 and 30 degrees

            image = torch.tensor(item["image"])
            image_noisy = torch.tensor(item["image_noisy"])

            item["image"] = rotation(image)
            item["image_noisy"] = rotation(image_noisy)

            return item

        data_train, data_valid = split_iterables(generate_random_data_for_test(
            num_samples=self.config_data["num_samples"], image_size=self.config_data["image_size"]),
            portions=self.config_data["split_portions"],
            shuffle=True)

        dataset_train_without_augmentation = Dataset.from_dict(data_train)
        dataset_train_with_augmentation = Dataset.from_dict(data_train).map(lambda x: augment(x))
        dataset_valid_without_augmentation = Dataset.from_dict(data_valid)
        dataset_valid_with_augmentation = Dataset.from_dict(data_valid).map(augment)

        return (dataset_train_with_augmentation,
                ValidationDatasetsDict(
                    datasets=(dataset_train_without_augmentation,
                              dataset_valid_with_augmentation,
                              dataset_valid_without_augmentation),
                    only_for_demo=(True, False, False),
                    names=("train-no-aug", "valid-with-aug", "valid-no-aug")
                ))

    def init_network(self) -> torch.nn.Module:
        class SomeSimpleCNN(torch.nn.Module):
            def __init__(self, num_ch: int):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, num_ch, 3, padding=1)
                self.conv2 = torch.nn.Conv2d(num_ch, num_ch, 3, padding=1)
                self.conv3 = torch.nn.Conv2d(num_ch, 3, 3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.conv1(x)
                x = torch.nn.ReLU()(x)
                x = self.conv2(x)
                x = torch.nn.ReLU()(x)
                x = self.conv3(x)

                return x

        network = SomeSimpleCNN(num_ch=self.config_network["num_ch"])

        return network

    def forward_pass(self, mini_batch: Dict[str, Any | torch.Tensor]) -> Dict[str, Any | torch.Tensor]:
        input_image = mini_batch["image_noisy"].to(self.device)
        target_image = mini_batch["image"].to(self.device)

        output_image = self.network(input_image)

        return {"output": output_image, "target": target_image,
                "gt_for_metrics": target_image.sigmoid().round(), "predictions_for_metrics": output_image.sigmoid()}

    def loss_function(self, *, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        criterion = torch.nn.BCEWithLogitsLoss()

        return criterion(output, target)

    def init_metrics(self) -> List[BaseMetricsClass] | None:
        metrics_class_binary_classification = BinaryClassificationMetrics(keyword_maps={
            "gt_for_metrics": "binary_ground_truth",
            "predictions_for_metrics": "prediction_probabilities"})

        metrics_class_image_reconstruction = ImageReconstructionMetrics(keyword_maps={
            "gt_for_metrics": "target_image",
            "predictions_for_metrics": "output_image"
        })

        metrics_classes_list = [metrics_class_binary_classification, metrics_class_image_reconstruction]
        return metrics_classes_list


class TrainingBaseSessionDynamicUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.path.exists(os.path.join(TEST_STORAGE_DIR)):
            shutil.rmtree(os.path.join(TEST_STORAGE_DIR))
        os.makedirs(TEST_STORAGE_DIR, exist_ok=True)
        # TODO: Create a temp folder in memory not disk

        cls.session = ExampleTrainingSessionClassDynamic(
            config=ExampleTrainingSessionClassDynamic.get_config(),
            runs_parent_dir=TEST_STORAGE_DIR,
            tag_postfix="dynamic"
        )

    @classmethod
    def tearDown(cls) -> None:
        cls.session.writer.close()

    def test_forward_and_loss_functions(self):
        self.assertIn("loss", self.session.value_logger_train.names)
        mini_batch = next(iter(self.session.dataloader_train))
        outs = self.session.forward_pass(mini_batch)

        sig = inspect.signature(self.session.loss_function)
        outs_for_loss = {k: v for k, v in outs.items() if k in sig.parameters}
        loss_tensor = self.session.loss_function(**outs_for_loss)

        self.assertIsInstance(loss_tensor, torch.Tensor)
        self.assertTrue(loss_tensor.requires_grad)
        self.assertIsInstance(self.session.get_loss_value(loss_tensor=loss_tensor), float)

    def test_infer_mini_batch_size(self):
        for mini_batch in self.session.dataloader_train:
            inferred_mini_batch_size = self.session.infer_mini_batch_size(mini_batch)
            self.assertLessEqual(inferred_mini_batch_size, self.session.config_session.mini_batch_size)

    def test_metrics_functionals(self):
        metrics_functionals_dict = self.session.metrics_functionals_dict
        mini_batch = next(iter(self.session.dataloader_train))
        outs = self.session.forward_pass(mini_batch)

        for metric_name, metric_functional in metrics_functionals_dict.items():
            sig = inspect.signature(metric_functional)
            outs_for_metric = {k: v for k, v in outs.items() if k in sig.parameters}
            metric_value = metric_functional(**outs_for_metric)
            self.assertIsInstance(metric_value, float)

    def test_custom_scalar_logging_layout_valid(self):
        layout = {
            'Loss': {
                'Loss (train vs val)': ['Multiline', ['training/loss/epochs',
                                                      'validation-valid-with-aug/loss/epochs']],
                'Loss valid (with and without aug)': ['Line', ['validation-valid-with-aug/loss/epochs',
                                                               'validation-valid-no-aug/loss/epochs']],

                'PSNR valid (with and without aug)': ['Line', ['validation-valid-with-aug/psnr/epochs',
                                                               'validation-valid-no-aug/psnr/epochs']],

            }
        }

        self.session.add_writer_custom_scalar_logging_layout(layout)

    def test_do_one_training_iteration(self):
        self.assertEqual(self.session.value_logger_train.average_of_epoch["loss"], 0.0)
        self.assertEqual(self.session.value_logger_train.average_overall["loss"], 0.0)
        self.assertEqual(self.session.progress_train.iter_current_epoch, 0)
        self.assertEqual(self.session.progress_train.epoch, 0)

        mini_batch = next(iter(self.session.dataloader_train))
        self.session.do_one_training_iteration(mini_batch)

        self.assertTrue(self.session.network.training)
        self.assertNotEqual(self.session.value_logger_train.average_of_epoch["loss"], 0.0)
        self.assertEqual(self.session.value_logger_train.average_overall["loss"],
                         self.session.value_logger_train.average_of_epoch["loss"])
        self.assertEqual(self.session.progress_train.iter_current_epoch, 1)
        self.assertEqual(self.session.progress_train.epoch, 0)

        self.session.value_logger_train.reset()

    def test_do_one_validation_iteration(self):
        for valid_dataset_name in self.session.datasets_valid_dict.names:
            mini_batch = next(iter(self.session.dataloader_valid_dict[valid_dataset_name]))
            self.session.do_one_validation_iteration(mini_batch, valid_dataset_name)
            self.assertFalse(self.session.network.training)
            self.assertNotEqual(self.session.value_logger_valid_dict[valid_dataset_name].average_of_epoch["loss"],
                                0.0)

            self.session.value_logger_valid_dict[valid_dataset_name].reset()

    def test_do_one_training_epoch(self):
        self.assertEqual(self.session.value_logger_train.average_of_epoch["loss"], 0.0)
        self.assertEqual(self.session.value_logger_train.average_overall["loss"], 0.0)
        self.assertEqual(self.session.progress_train.iter_current_epoch, 0)
        self.assertEqual(self.session.progress_train.epoch, 0)

        self.session.do_one_training_epoch()

        self.assertNotEqual(self.session.value_logger_train.average_overall["loss"], 0.0)
        self.assertEqual(self.session.progress_train.iter_current_epoch, 0)
        self.assertEqual(self.session.progress_train.epoch, 1)

        self.session.value_logger_train.reset()

    def test_do_one_validation_epoch(self):
        for valid_dataset_name in self.session.datasets_valid_dict.names:
            self.session.do_one_validation_epoch(valid_dataset_name)

            self.assertNotEqual(self.session.value_logger_valid_dict[valid_dataset_name].average_overall["loss"], 0.0)
            self.assertEqual(self.session.progress_valid_dict[valid_dataset_name].iter_current_epoch, 0)
            self.assertEqual(self.session.progress_valid_dict[valid_dataset_name].epoch, 1)

            self.session.value_logger_valid_dict[valid_dataset_name].reset()

    def test_append_optional_hparams(self):
        optional_hparams_dict = {"num_ch": self.session.config_network["num_ch"]}
        self.session.append_hparams_dict(optional_hparams_dict)

    def test_do_training(self):
        self.session.train()


if __name__ == "__main__":
    unittest.main()
