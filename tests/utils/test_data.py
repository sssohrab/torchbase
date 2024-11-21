from torchbase.utils.data import random_split_iterable, split_to_train_valid_test
from torchbase.utils.data import TypedDict, TypedDictIterable

import unittest
from io import StringIO

from enum import Enum
from dataclasses import dataclass


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


if __name__ == "__main__":
    unittest.main()
