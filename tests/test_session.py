import torch

from torchbase.session import TrainingBaseSession
from torchbase.session import SAVED_RNG_NAME
from torchbase.utils.data import ValidationDatasetsDict, split_iterables

from datasets import Dataset

import unittest

from typing import Dict, Tuple
import os
import random
import shutil
import time
import json

TEST_STORAGE_DIR = os.path.join(os.path.split(os.path.abspath(__file__))[0], "storage")
os.makedirs(TEST_STORAGE_DIR, exist_ok=True)


class ExampleTrainingSessionClass(TrainingBaseSession):
    @staticmethod
    def get_config() -> Dict:
        config = {
            "session": {
                "device_name": "cpu",
                "num_epochs": 5,
                "mini_batch_size": 4,
                "learning_rate": 0.001,
                "weight_decay": 1e-6,
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


class TrainingBaseSessionInitializationUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.path.exists(os.path.join(TEST_STORAGE_DIR)):
            shutil.rmtree(os.path.join(TEST_STORAGE_DIR))
        os.makedirs(TEST_STORAGE_DIR, exist_ok=True)
        # TODO: Create a temp folder in memory not disk

        cls.session_fresh_run_fresh_network = ExampleTrainingSessionClass(
            config=ExampleTrainingSessionClass.get_config(),
            runs_parent_dir=TEST_STORAGE_DIR,
            create_run_dir_afresh=True,
            source_run_dir_tag=None
        )

        cls.mock_train_steps_and_save_network_and_optimizer()

        time.sleep(1)  # To avoid creating the same tag again.

        cls.session_existing_run = ExampleTrainingSessionClass(
            config=ExampleTrainingSessionClass.get_config(),
            runs_parent_dir=TEST_STORAGE_DIR,
            create_run_dir_afresh=False,
            source_run_dir_tag=os.path.split(cls.session_fresh_run_fresh_network.run_dir)[-1]
        )

        cls.session_fresh_run_pretrained_network = ExampleTrainingSessionClass(
            config=ExampleTrainingSessionClass.get_config(),
            runs_parent_dir=TEST_STORAGE_DIR,
            create_run_dir_afresh=True,
            source_run_dir_tag=os.path.split(cls.session_fresh_run_fresh_network.run_dir)[-1]
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

    def test_instantiate_session_with_fresh_run_fresh_network(self):
        self.assertIsNotNone(self.session_fresh_run_fresh_network)
        self.assertIsInstance(self.session_fresh_run_fresh_network, TrainingBaseSession)

    def test_instantiate_session_with_existing_run(self):
        self.assertIsNotNone(self.session_existing_run)
        self.assertIsInstance(self.session_existing_run, TrainingBaseSession)

    def test_instantiate_session_with_fresh_run_pretrained_network(self):
        self.assertIsNotNone(self.session_fresh_run_pretrained_network)
        self.assertIsInstance(self.session_fresh_run_pretrained_network, TrainingBaseSession)

    def test_datasets_random_access(self):
        datasets = ([self.session_fresh_run_fresh_network.dataset_train]
                    + list(self.session_fresh_run_fresh_network.datasets_valid_dict.datasets))
        for dataset in datasets:
            idx = random.randint(0, dataset.__len__() - 1)
            for expected_keys in ["inputs", "labels"]:
                self.assertIn(expected_keys, dataset[idx])

    def test_saved_random_states_replicability(self):
        dataset_initial = self.session_fresh_run_fresh_network.dataset_train
        dataset_replicated = self.session_existing_run.dataset_train
        self.assertEqual(dataset_initial.data, dataset_replicated.data)

    def test_existing_run_dir_but_no_run_tag_specified(self):
        with self.assertRaises(ValueError):
            ExampleTrainingSessionClass(config=ExampleTrainingSessionClass.get_config(),
                                        runs_parent_dir=TEST_STORAGE_DIR,
                                        create_run_dir_afresh=False,
                                        source_run_dir_tag="Some-non-existing-tag")

        # TODO: When networks loading is implemented, to test with a wrong source-dir.

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
        wrong_config = ExampleTrainingSessionClass.get_config()
        wrong_config["network"]["architecture"] = "SomeMistakenlyHeldNetworkName"
        with self.assertRaises(TypeError):
            time.sleep(1)  # To avoid creating the same tag again.
            ExampleTrainingSessionClass(config=wrong_config, runs_parent_dir=TEST_STORAGE_DIR)

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
            ExampleTrainingSessionClass(
                config=ExampleTrainingSessionClass.get_config(),
                runs_parent_dir=TEST_STORAGE_DIR,
                create_run_dir_afresh=False,
                source_run_dir_tag=os.path.split(source_session.run_dir)[-1]
            )


if __name__ == "__main__":
    unittest.main()
