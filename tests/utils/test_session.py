from torchbase.utils.session import TrainingConfigSessionDict
from torchbase.utils.session import is_custom_scalar_logging_layout_valid
from torchbase.utils.session import RandomnessGeneratorStates

import unittest

import torch
import numpy as np
import random

import json
import tempfile
import os


class CustomScalarLoggingLayoutValidityUnitTest(unittest.TestCase):
    def test_custom_scalar_logging_layout_valid(self):
        layout = {
            'Loss': {
                'Loss (train vs val)': ['Multiline', ['training/loss/epochs', 'validation-valid/loss/epochs']],
                'Loss valid (with and without aug)': ['Line', ['validation-valid/loss/epochs',
                                                               'validation-valid_with_augmentation/loss/epochs']],
                'Recall valid (with and without aug)': ['Line', ['validation-valid/recall/epochs',
                                                                 'validation-valid_with_augmentation/recall/epochs'
                                                                 ]]
            }
        }

        self.assertTrue(is_custom_scalar_logging_layout_valid(layout,
                                                              validation_dataset_names=("valid",
                                                                                        "valid_with_augmentation"),
                                                              metric_names=("precision", "recall")))

    def test_custom_scalar_logging_layout_invalid_wrong_set_name(self):
        layout = {
            'Loss': {
                'Loss (train vs val)': ['Multiline', ['train/loss/epochs', 'validation-valid/loss/epochs']],
                'Loss valid (with and without aug)': ['Line', ['validation-valid/loss/epochs',
                                                               'validation-valid_with_augmentation/loss/epochs']],

            }
        }

        self.assertFalse(is_custom_scalar_logging_layout_valid(layout,
                                                               validation_dataset_names=("valid",
                                                                                         "valid_with_augmentation"),
                                                               metric_names=("precision", "recall")))

    def test_custom_scalar_logging_layout_invalid_non_existing_metric(self):
        layout = {
            'Loss': {
                'Loss (train vs val)': ['Multiline', ['training/loss/epochs', 'validation-valid/loss/epochs']],
                'Loss valid (with and without aug)': ['Line', ['validation-valid/loss/epochs',
                                                               'validation-valid_with_augmentation/loss/epochs']],
                'AUC valid (with and without aug)': ['Line', ['validation-valid/AUC/epochs',
                                                              'validation-valid_with_augmentation/AUC/epochs'
                                                              ]]
            }
        }

        self.assertFalse(is_custom_scalar_logging_layout_valid(layout,
                                                               validation_dataset_names=("valid",
                                                                                         "valid_with_augmentation"),
                                                               metric_names=("precision", "recall")))


class TrainingConfigSessionDictUnitTest(unittest.TestCase):

    def test_correct_input(self):
        config_session = {
            "device_name": "cpu",
            "num_epochs": 10,
            "mini_batch_size": 3,
            "learning_rate": 0.001,
            "weight_decay": 1e-6,
            "dataloader_num_workers": 0,
            "loss_function_params": None,
        }
        config_session = TrainingConfigSessionDict(config_session)
        self.assertTrue(config_session.is_valid())

    def test_missing_necessary_field(self):
        config_session = {
            "device_name": "cpu",
            "num_epochs": 10,
            "mini_batch_size": 3,
            "weight_decay": 1e-6,
            "dataloader_num_workers": 0,
            "loss_function_params": None,
        }
        with self.assertRaises(ValueError):
            TrainingConfigSessionDict(config_session)


class RandomnessGeneratorStatesUnitTest(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.rng_states_path = os.path.join(self.temp_dir.name, "rng_states.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_load_apply(self):
        rng1 = RandomnessGeneratorStates()
        rng1.save(self.rng_states_path)

        torch_rand1 = torch.rand(3).tolist()
        np_rand1 = np.random.rand(3).tolist()
        py_rand1 = [random.random() for _ in range(3)]

        torch.manual_seed(999)
        np.random.seed(999)
        random.seed(999)

        rng2 = RandomnessGeneratorStates.load(self.rng_states_path)
        rng2.apply()

        torch_rand2 = torch.rand(3).tolist()
        np_rand2 = np.random.rand(3).tolist()
        py_rand2 = [random.random() for _ in range(3)]

        self.assertEqual(torch_rand1, torch_rand2)
        self.assertEqual(np_rand1, np_rand2)
        self.assertEqual(py_rand1, py_rand2)

    def test_fresh_state_is_different(self):
        rng_states_path_2 = os.path.join(self.temp_dir.name, "rng_states_2.json")

        rng1 = RandomnessGeneratorStates()

        # Just consuming some randomness before creating the next one:
        torch.rand(10)
        np.random.rand(10)
        random.random()

        rng2 = RandomnessGeneratorStates()

        rng1.save(self.rng_states_path)
        rng2.save(rng_states_path_2)

        loaded_rng1 = RandomnessGeneratorStates.load(self.rng_states_path)
        loaded_rng2 = RandomnessGeneratorStates.load(rng_states_path_2)

        loaded_rng1.apply()
        torch_rand1 = torch.rand(1).item()
        np_rand1 = np.random.rand()
        py_rand1 = random.random()

        loaded_rng2.apply()
        torch_rand2 = torch.rand(1).item()
        np_rand2 = np.random.rand()
        py_rand2 = random.random()

        self.assertNotAlmostEqual(torch_rand1, torch_rand2, places=7)
        self.assertNotAlmostEqual(np_rand1, np_rand2, places=7)
        self.assertNotAlmostEqual(py_rand1, py_rand2, places=7)

    def test_random_state_tuple_structure(self):
        rng1 = RandomnessGeneratorStates()
        rng1.save(self.rng_states_path)

        rng2 = RandomnessGeneratorStates.load(self.rng_states_path)
        self.assertIsInstance(rng2.random_state, tuple)
        self.assertIsInstance(rng2.random_state[0], int)
        self.assertIsInstance(rng2.random_state[1], tuple)
        self.assertIsInstance(rng2.random_state[2], (type(None), float))

    def test_numpy_state_structure(self):
        rng1 = RandomnessGeneratorStates()
        rng1.save(self.rng_states_path)

        rng2 = RandomnessGeneratorStates.load(self.rng_states_path)
        self.assertIsInstance(rng2.numpy_state, tuple)
        self.assertIsInstance(rng2.numpy_state[0], str)
        self.assertIsInstance(rng2.numpy_state[1], np.ndarray)
        self.assertEqual(rng2.numpy_state[1].dtype, np.uint32)
        self.assertIsInstance(rng2.numpy_state[2], int)
        self.assertIsInstance(rng2.numpy_state[3], int)
        self.assertIsInstance(rng2.numpy_state[4], float)

    def test_reproducibility_across_runs(self):
        rng1 = RandomnessGeneratorStates()
        rng1.save(self.rng_states_path)

        random_numbers1 = [random.random() for _ in range(5)]
        torch_tensor1 = torch.rand(5).tolist()
        numpy_array1 = np.random.rand(5).tolist()

        # Simulate a different execution session:
        torch.manual_seed(123)
        np.random.seed(123)
        random.seed(123)

        rng2 = RandomnessGeneratorStates.load(self.rng_states_path)
        rng2.apply()

        random_numbers2 = [random.random() for _ in range(5)]
        torch_tensor2 = torch.rand(5).tolist()
        numpy_array2 = np.random.rand(5).tolist()

        self.assertEqual(random_numbers1, random_numbers2)
        self.assertEqual(torch_tensor1, torch_tensor2)
        self.assertEqual(numpy_array1, numpy_array2)


if __name__ == "__main__":
    unittest.main()
