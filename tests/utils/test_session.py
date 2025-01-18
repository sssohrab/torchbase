from torchbase.utils.session import TrainingConfigSessionDict
from torchbase.utils.session import is_custom_scalar_logging_layout_valid

import unittest


class CustomScalarLoggingLayoutValidityUnitTest(unittest.TestCase):
    def test_custom_scalar_logging_layout_valid(self):
        layout = {
            'Loss': {
                'Loss (train vs val)': ['Multiline', ['training/loss/epochs', 'validation-valid/loss/epochs']],
                'Loss valid (with and without aug)': ['Line', ['validation-valid/loss/epochs',
                                                               'validation-valid_with_augmentation/loss/epochs']],
            }
        }

        self.assertTrue(is_custom_scalar_logging_layout_valid(layout,
                                                              validation_dataset_names=("valid",
                                                                                        "valid_with_augmentation"),
                                                              metric_names=("precision", "recall")))

    def test_custom_scalar_logging_layout_invalid(self):
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
        config_session = TrainingConfigSessionDict(config_session)
        self.assertFalse(config_session.is_valid())


if __name__ == "__main__":
    unittest.main()
