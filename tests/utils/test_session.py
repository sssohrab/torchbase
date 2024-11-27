from torchbase.utils.session import TrainingConfigSessionDict

import unittest


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
