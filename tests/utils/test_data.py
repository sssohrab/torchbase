from torchbase.utils.data import split_iterables, split_to_train_valid_test
from torchbase.utils.data import TypedDict, TypedDictIterable
from torchbase.utils.data import ValidationDatasetsDict

import unittest

from datasets import Dataset

from enum import Enum
from dataclasses import dataclass

from io import StringIO
import random


class SplittingTests(unittest.TestCase):

    def setUp(self):
        self.data_list = ["sample{}".format(i) for i in range(10)]
        self.data_file = StringIO("\n".join(self.data_list) + "\n")
        self.dictionary_of_iterables = {
            "data": self.data_list,
            "files": self.data_file,
            "indices": tuple([i for i in range(10)]),
            "IDs": ["ID-{}".format(i) for i in range(10)]
        }

    def test_list_input(self):
        split_1, split_2, split_3 = split_iterables(self.data_list, (0.6, 0.2, 0.2))

        self.assertEqual(len(split_1), 6)
        self.assertEqual(len(split_2), 2)
        self.assertEqual(len(split_3), 2)
        self.assertEqual(set(split_1 + split_2 + split_3), set(self.data_list))

    def test_list_input_without_shuffle(self):
        split_1, split_2, split_3 = split_iterables(self.data_list, (0.6, 0.2, 0.2), shuffle=False)

        self.assertEqual(len(split_1), 6)
        self.assertEqual(len(split_2), 2)
        self.assertEqual(len(split_3), 2)
        self.assertEqual(split_1 + split_2 + split_3, self.data_list)

    def test_file_input(self):
        split_1, split_2 = split_iterables(self.data_file, (1.0, 4.0))

        self.assertEqual(len(split_1), 2)
        self.assertEqual(len(split_2), 8)
        self.assertEqual(set(split_1 + split_2), set(self.data_list))

    def test_dictionary_input(self):
        split_1, split_2 = split_iterables(self.dictionary_of_iterables, (0.8, 0.2))

        self.assertEqual(sorted(list(split_1.keys())), sorted(list(self.dictionary_of_iterables.keys())))

        for v in split_1.values():
            self.assertEqual(len(v), 8)

        for i in range(8):
            for v in split_1.values():
                self.assertTrue(str(v[i]).endswith(str(split_1["indices"][i])))

        for v in split_2.values():
            self.assertEqual(len(v), 2)

    def test_invalid_portions(self):
        with self.assertRaises(ValueError):
            split_iterables(self.data_list, (0.6, 0.2, -0.2))

    def test_empty_input(self):
        empty_data = []
        split_1, split_2 = split_iterables(empty_data, (2.0, 3.0))

        self.assertEqual(len(split_1), 0)
        self.assertEqual(len(split_2), 0)

    def test_split_to_train_valid_test_valid(self):
        train, valid, test = split_iterables(self.data_file, (0.7, 0.15, 0.15))
        self.assertEqual(len(train), 7)
        self.assertEqual(len(valid), 1)
        self.assertEqual(len(test), 2)
        self.assertEqual(set(train + valid + test), set(self.data_list))

    def test_split_to_train_valid_test_invalid(self):
        with self.assertRaises(ValueError):
            train, valid = split_to_train_valid_test(self.data_file, (0.7, 0.3))

    def test_split_to_train_valid_test_for_dictionary(self):
        train_dict, valid_dict, test_dict = split_iterables(self.dictionary_of_iterables, (0.7, 0.15, 0.15))
        for v in train_dict.values():
            self.assertEqual(len(v), 7)
        for v in valid_dict.values():
            self.assertEqual(len(v), 1)
        for v in test_dict.values():
            self.assertEqual(len(v), 2)


class ExampleCategoricalDataTypeGender(Enum):
    MALE = 0
    FEMALE = 1
    NON_BINARY = -1

    @staticmethod
    def from_string(value: str | float):
        if str(value).lower() in [0, "man", "male", "guy", "homme", "gars"]:
            return ExampleCategoricalDataTypeGender.MALE
        elif str(value).lower() in [1, "woman", "female", "femme", "meuf"]:
            return ExampleCategoricalDataTypeGender.FEMALE
        elif str(value).lower() in [-1, "non-binary", "nonbinary"]:
            return ExampleCategoricalDataTypeGender.NON_BINARY
        else:
            raise ValueError("Could not infer gender from text.")

    def __str__(self):
        if self.value == 0:
            return "MALE"
        elif self.value == 1:
            return "FEMALE"
        elif self.value == -1:
            return "NON_BINARY"
        else:
            raise ValueError


@dataclass
class ExampleNestedDataClass:
    email: str
    phone_number: str
    job_title: str | None

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            email=data.get("email"),
            phone_number=data.get("phone_number"),
            job_title=data.get("job_title"),
        )


class TypedDictUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.typed_dict = TypedDict({
            "name": str,
            "age": int,
            "gender": ExampleCategoricalDataTypeGender,
            "details": ExampleNestedDataClass | None
        })

    def test_correct_types(self):
        input_dict = {
            "name": "John Foo",
            "age": 43,
            "gender": ExampleCategoricalDataTypeGender.MALE,
            "details": ExampleNestedDataClass.from_dict({"email": "john@foo.him",
                                                         "phone_number": "+41-22-22",
                                                         "job_title": "Lead Expert of Staff"})
        }

        output_dict = self.typed_dict(input_dict)
        self.assertEqual(input_dict, output_dict)

    def test_incorrect_type(self):
        input_dict = {
            "name": "Jane Foo",
            "age": "43",
            "gender": ExampleCategoricalDataTypeGender.FEMALE,
            "details": None
        }
        with self.assertRaises(TypeError):
            self.typed_dict(input_dict)

    def test_allowed_none(self):
        input_dict = {
            "name": "John Foo",
            "age": 43,
            "gender": ExampleCategoricalDataTypeGender.from_string("man"),
            "details": None
        }

        output_dict = self.typed_dict(input_dict)
        self.assertEqual(input_dict, output_dict)

    def test_not_allowed_none(self):
        input_dict = {
            "name": "John Foo",
            "age": 43,
            "gender": None,
            "details": ExampleNestedDataClass.from_dict({"email": "john@foo.him",
                                                         "phone_number": "+41-22-22",
                                                         "job_title": "Lead Expert of Staff"})
        }
        with self.assertRaises(TypeError):
            self.typed_dict(input_dict)


class TypedDictIterableUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.typed_dict_iterable = TypedDictIterable({
            "name": str,
            "age": int,
            "gender": ExampleCategoricalDataTypeGender,
            "details": ExampleNestedDataClass | None

        })

    def test_correct_types(self):
        input_dict = {
            "name": ["John Foo", "Jim Bar"],
            "age": [43, 22],
            "gender": [ExampleCategoricalDataTypeGender.MALE, ExampleCategoricalDataTypeGender.NON_BINARY],
            "details": [ExampleNestedDataClass.from_dict({"email": "john@foo.him",
                                                          "phone_number": "+41-22-22",
                                                          "job_title": "Lead Expert of Staff"}), None]
        }

        output_dict = self.typed_dict_iterable(input_dict)
        self.assertEqual(input_dict, output_dict)

    def test_incorrect_type(self):
        input_dict = {
            "name": ["John Foo", "Jim Bar"],
            "age": [43, "22"],
            "gender": [ExampleCategoricalDataTypeGender.MALE, ExampleCategoricalDataTypeGender.NON_BINARY],
            "details": ["None", None]
        }
        with self.assertRaises(TypeError):
            self.typed_dict_iterable(input_dict)

    def test_incorrect_iterable(self):
        input_dict = {
            "name": ["John Foo", "Jim Bar"],
            "age": "AZ",
            "gender": [ExampleCategoricalDataTypeGender.MALE, ExampleCategoricalDataTypeGender.NON_BINARY],
            "details": [None, None]
        }
        with self.assertRaises(TypeError):
            self.typed_dict_iterable(input_dict)

    def test_no_iterable(self):
        input_dict = {
            "name": ["John Foo", "Jim Bar"],
            "age": 23,
            "gender": [ExampleCategoricalDataTypeGender.MALE, ExampleCategoricalDataTypeGender.NON_BINARY],
            "details": [None, None]
        }
        with self.assertRaises(TypeError):
            self.typed_dict_iterable(input_dict)

    def test_variable_length_iterables(self):
        input_dict = {
            "name": ["John Foo", "Jim Bar"],
            "age": [43, 22, 13],
            "gender": [ExampleCategoricalDataTypeGender.MALE, ExampleCategoricalDataTypeGender.NON_BINARY],
            "details": [None]
        }
        with self.assertRaises(ValueError):
            self.typed_dict_iterable(input_dict)

    def test_data_attribute_valid_set_and_get(self):
        input_dict = {
            "name": ["John Foo", "Jim Bar"],
            "age": [43, 22],
            "gender": [ExampleCategoricalDataTypeGender.MALE, ExampleCategoricalDataTypeGender.NON_BINARY],
            "details": [None, None]
        }

        self.typed_dict_iterable.data = input_dict
        self.assertEqual(input_dict, self.typed_dict_iterable.data)

    def test_data_attribute_invalid_set_and_get(self):
        input_dict = {
            "name": ["John Foo", "Jim Bar"],
            "age": [43],
            "gender": [ExampleCategoricalDataTypeGender.MALE, ExampleCategoricalDataTypeGender.NON_BINARY],
            "details": [None, None]
        }

        with self.assertRaises(ValueError):
            self.typed_dict_iterable.data = input_dict


class ValidationDatasetsDictUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset_valid_1 = Dataset.from_dict({"data": [random.randint(0, 100) for _ in range(10)]})
        cls.dataset_valid_2 = Dataset.from_dict({"data": [random.randint(0, 100) for _ in range(10)]})
        cls.dataset_valid_3 = Dataset.from_dict({"data": [random.randint(0, 100) for _ in range(10)]})

    def test_correct_input(self):
        datasets = (self.dataset_valid_1, self.dataset_valid_2, self.dataset_valid_3)
        only_for_demo = (True, False, True)
        names = ("train_without_augmentation", "valid_with_augmentation", "valid_without_augmentation")
        config = ValidationDatasetsDict(datasets=datasets, only_for_demo=only_for_demo, names=names)
        self.assertTrue(config.is_valid())

    def test_incorrect_type_datasets(self):
        incorrect_datasets = (123, self.dataset_valid_2, self.dataset_valid_3)
        only_for_demo = (True, False, True)
        names = ("train", "valid_1", "valid_2")
        config = ValidationDatasetsDict(datasets=incorrect_datasets, only_for_demo=only_for_demo, names=names)
        self.assertFalse(config.is_valid())

    def test_incorrect_type_only_for_demo(self):
        datasets = (self.dataset_valid_1, self.dataset_valid_2)
        incorrect_flags = ("yes", "no")
        names = ("train", "valid")
        config = ValidationDatasetsDict(datasets=datasets, only_for_demo=incorrect_flags, names=names)
        self.assertFalse(config.is_valid())

    def test_mismatched_length(self):
        datasets = (self.dataset_valid_1, self.dataset_valid_2)
        flags = (True,)
        names = ("train", "valid_1", "valid_2")
        config = ValidationDatasetsDict(datasets=datasets, only_for_demo=flags, names=names)
        self.assertFalse(config.is_valid())


if __name__ == "__main__":
    unittest.main()
