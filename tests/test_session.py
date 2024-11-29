from torchbase.session import TrainingBaseSession
from torchbase.utils.data import ValidationDatasetsDict, split_iterables

from datasets import Dataset

import unittest

from typing import Dict, List, Tuple, Any, Callable
import os
import random
import shutil

TEST_STORAGE_DIR = os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0], "storage")
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
                "inputs": [random.randint(-10, 10) for _ in range(20)],
                "labels": [random.randint(0, 1) for _ in range(20)],
                "split_portions": (0.8, 0.2)
            },
            "metrics": {},
            "network": {}
        }

        return config

    def init_datasets(self) -> Tuple[Dataset, ValidationDatasetsDict]:
        data_train, data_valid = split_iterables(
            {k: v for k, v in self.config_data.items() if k in ["inputs", "labels"]},
            portions=self.config_data["split_portions"])

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
                    names=("train-without-aug", "valid-with-aug", "valid-without-aug")
                ))


class TrainingBaseSessionInitializationUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.path.exists(os.path.join(TEST_STORAGE_DIR, "runs", "train")):
            shutil.rmtree(os.path.join(TEST_STORAGE_DIR, "runs", "train"))
        os.makedirs(TEST_STORAGE_DIR, exist_ok=True)

        cls.session_fresh_run_fresh_network = ExampleTrainingSessionClass(
            config=ExampleTrainingSessionClass.get_config(),
            create_run_dir_afresh=True,
            source_run_dir_tag=None
        )

    def test_instantiate_session_with_fresh_run_fresh_network(self):
        self.assertIsNotNone(self.session_fresh_run_fresh_network)
        self.assertIsInstance(self.session_fresh_run_fresh_network, TrainingBaseSession)

    def test_datasets_random_access(self):
        datasets = ([self.session_fresh_run_fresh_network.dataset_train]
                    + list(self.session_fresh_run_fresh_network.datasets_valid_dict.datasets))
        for dataset in datasets:
            idx = random.randint(0, dataset.__len__())
            for expected_keys in ["inputs", "labels"]:
                self.assertIn(expected_keys, dataset[idx])
