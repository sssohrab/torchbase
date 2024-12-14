from torchbase.utils.logger import ProgressManager
from torchbase.utils.logger import ValuesLogger

import unittest

import tempfile
import os


class ProgressAndValuesLoggerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.names = ['loss', 'accuracy']
        self.progress_manager = ProgressManager()
        self.values_logger = ValuesLogger(names=self.names, progress_manager=self.progress_manager)

        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_increment_iter(self):
        self.progress_manager.increment_iter(mini_batch_size=10)
        self.assertEqual(self.progress_manager.iter_current_epoch, 1)
        self.assertEqual(self.progress_manager.samples_current_epoch, 10)
        self.assertEqual(self.progress_manager.samples_total, 10)

        with self.assertRaises(ValueError):
            self.progress_manager.increment_iter(mini_batch_size=-5)

    def test_increment_epoch(self):
        self.progress_manager.increment_iter(mini_batch_size=10)
        self.progress_manager.increment_epoch()
        self.assertEqual(self.progress_manager.epoch, 1)
        self.assertEqual(self.progress_manager.iter_current_epoch, 0)
        self.assertEqual(self.progress_manager.samples_current_epoch, 0)
        # TODO: Test that overall values are not reset.

    def test_values_logger_update_and_reset(self):
        self.progress_manager.increment_iter(mini_batch_size=100)
        values_dict = {'loss': 0.25, 'accuracy': 0.85}
        self.values_logger.update(values_dict)
        self.assertEqual(self.values_logger.average_of_epoch['loss'], 0.25)
        self.assertEqual(self.values_logger.average_of_epoch['accuracy'], 0.85)
        self.assertEqual(self.values_logger.average_overall['loss'], 0.25)
        self.assertEqual(self.values_logger.average_overall['accuracy'], 0.85)

        self.values_logger.reset_epoch()
        self.assertEqual(self.values_logger.average_of_epoch['loss'], 0.0)
        self.assertEqual(self.values_logger.average_of_epoch['accuracy'], 0.0)

    def test_invalid_values_update(self):
        with self.assertRaises(TypeError):
            self.values_logger.update({'loss': 'high', 'accuracy': 0.90})
        with self.assertRaises(ValueError):
            self.values_logger.update({'loss': 0.25})
        with self.assertRaises(ValueError):
            self.values_logger.update({'loss': 0.25, 'accuracy': 0.90, 'extra_metric': 0.1})  # extra keys

    def test_epoch_reset_preserves_overall_averages(self):
        self.progress_manager.increment_iter(mini_batch_size=100)
        values_dict_first = {'loss': 0.25, 'accuracy': 0.85}
        self.values_logger.update(values_dict_first)

        self.progress_manager.increment_iter(mini_batch_size=200)
        values_dict_second = {'loss': 0.35, 'accuracy': 0.90}
        self.values_logger.update(values_dict_second)

        overall_loss = self.values_logger.average_overall['loss']
        overall_accuracy = self.values_logger.average_overall['accuracy']

        self.assertEqual(overall_loss, (0.25 * 100 + 0.35 * 200) / (100 + 200))
        self.assertEqual(overall_accuracy, (0.85 * 100 + 0.90 * 200) / (100 + 200))

        self.values_logger.reset_epoch()
        self.progress_manager.increment_epoch()

        self.assertEqual(self.values_logger.average_overall['loss'], overall_loss)
        self.assertEqual(self.values_logger.average_overall['accuracy'], overall_accuracy)

        self.assertEqual(self.values_logger.average_of_epoch['loss'], 0.0)
        self.assertEqual(self.values_logger.average_of_epoch['accuracy'], 0.0)

    def test_running_average_equivalence(self):
        initial_samples = 50
        additional_samples = 150
        initial_values = {'loss': 0.20, 'accuracy': 0.80}
        additional_values = {'loss': 0.30, 'accuracy': 0.85}

        self.progress_manager.increment_iter(mini_batch_size=initial_samples)
        self.values_logger.update(initial_values)

        self.progress_manager.increment_iter(mini_batch_size=additional_samples)
        self.values_logger.update(additional_values)

        expected_loss = (initial_values['loss'] * initial_samples + additional_values['loss'] * additional_samples) / \
                        (initial_samples + additional_samples)
        expected_accuracy = (initial_values['accuracy'] * initial_samples + additional_values[
            'accuracy'] * additional_samples) / (initial_samples + additional_samples)

        computed_loss = self.values_logger.average_overall['loss']
        computed_accuracy = self.values_logger.average_overall['accuracy']

        self.assertAlmostEqual(computed_loss, expected_loss, places=4)
        self.assertAlmostEqual(computed_accuracy, expected_accuracy, places=4)

    def test_disk_save_and_load_progress_manager(self):
        self.progress_manager.increment_iter(mini_batch_size=3)
        self.progress_manager.increment_iter(mini_batch_size=2)

        states_json_path = os.path.join(self.temp_dir.name, "progress_manager.json")
        self.progress_manager.serialize_to_disk(states_json_path)
        loaded_progress_manager = ProgressManager()
        loaded_progress_manager.set_fields_from_disk(states_json_path)

        self.assertEqual(loaded_progress_manager.samples_total, 5)

    def test_disk_save_and_load_values_logger(self):
        self.progress_manager.increment_iter(mini_batch_size=100)
        values_dict = {'loss': 0.25, 'accuracy': 0.85}
        self.values_logger.update(values_dict)

        states_json_path = os.path.join(self.temp_dir.name, "values_logger_manager.json")
        self.values_logger.serialize_to_disk(states_json_path)

        loaded_values_logger = ValuesLogger(names=self.names, progress_manager=self.progress_manager)
        loaded_values_logger.set_state_values_from_disk(states_json_path)

        self.assertEqual(loaded_values_logger.average_of_epoch['loss'], 0.25)
        self.assertEqual(loaded_values_logger.average_of_epoch['accuracy'], 0.85)
        self.assertEqual(loaded_values_logger.average_overall['loss'], 0.25)
        self.assertEqual(loaded_values_logger.average_overall['accuracy'], 0.85)


if __name__ == '__main__':
    unittest.main()
