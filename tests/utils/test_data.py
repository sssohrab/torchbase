from torchbase.utils.data import random_split_iterable, split_to_train_valid_test
import unittest
from io import StringIO


class RandomSplittingUnitTest(unittest.TestCase):

    def setUp(self):
        self.data_list = ["sample{}".format(i + 1) for i in range(10)]
        self.data_file = StringIO("\n".join(self.data_list) + "\n")

    def test_list_input(self):
        split_1, split_2, split_3 = random_split_iterable(self.data_list, (0.6, 0.2, 0.2))

        self.assertEqual(len(split_1), 6)
        self.assertEqual(len(split_2), 2)
        self.assertEqual(len(split_3), 2)
        self.assertEqual(set(split_1 + split_2 + split_3), set(self.data_list))

    def test_file_input(self):
        split_1, split_2 = random_split_iterable(self.data_file, (1.0, 4.0))

        self.assertEqual(len(split_1), 2)
        self.assertEqual(len(split_2), 8)
        self.assertEqual(set(split_1 + split_2), set(self.data_list))

    def test_invalid_portions(self):
        with self.assertRaises(ValueError):
            random_split_iterable(self.data_list, (0.6, 0.2, -0.2))

    def test_empty_input(self):
        empty_data = []
        split_1, split_2 = random_split_iterable(empty_data, (2.0, 3.0))

        self.assertEqual(len(split_1), 0)
        self.assertEqual(len(split_2), 0)

    def test_split_to_train_valid_test_valid(self):
        train, valid, test = random_split_iterable(self.data_file, (0.7, 0.15, 0.15))
        self.assertEqual(len(train), 7)
        self.assertEqual(len(valid), 1)
        self.assertEqual(len(test), 2)
        self.assertEqual(set(train + valid + test), set(self.data_list))

    def test_split_to_train_valid_test_invalid(self):
        with self.assertRaises(ValueError):
            train, valid = split_to_train_valid_test(self.data_file, (0.7, 0.3))


if __name__ == "__main__":
    unittest.main()
