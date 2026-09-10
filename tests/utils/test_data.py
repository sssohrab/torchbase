from torchbase.utils.data import split_iterables, split_to_train_valid_test
from torchbase.utils.data import TypedDict, TypedDictIterable
from torchbase.utils.data import ValidationDatasetsDict

import unittest

from datasets import Dataset

from enum import Enum
from dataclasses import dataclass

from io import StringIO
from itertools import product
import random


class SizedOneShotIterator:
    def __init__(self):
        self.values = iter([1, 2])
        self.reads = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.reads += 1
        return next(self.values)

    def __len__(self):
        return 2


class UnsizedIterable:
    def __init__(self):
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return iter([1, 2])


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

    def test_empty_dictionary_and_empty_columns(self):
        self.assertEqual(split_iterables({}, (0.5, 0.5)), ({}, {}))
        self.assertEqual(split_iterables({"x": [], "y": ()}, (0.5, 0.5)),
                         ({"x": [], "y": []}, {"x": [], "y": []}))
        self.assertEqual(split_to_train_valid_test({}), ({}, {}, {}))

    def test_empty_dictionary_still_validates_portions(self):
        for portions, error in (((), TypeError), ((0.0, 0.0), ValueError), ((1.0, -0.5), ValueError)):
            with self.subTest(portions=portions), self.assertRaises(error):
                split_iterables({}, portions)

    def test_generators_are_materialized_without_losing_items(self):
        self.assertEqual(split_iterables((i for i in range(6)), (0.5, 0.5), shuffle=False),
                         ([0, 1, 2], [3, 4, 5]))
        values = {"x": (i for i in range(6)), "y": (i + 10 for i in range(6))}
        self.assertEqual(split_iterables(values, (0.5, 0.5), shuffle=False),
                         ({"x": [0, 1, 2], "y": [10, 11, 12]},
                          {"x": [3, 4, 5], "y": [13, 14, 15]}))

    def test_shuffled_generator_columns_stay_aligned(self):
        parts = split_iterables({"x": iter(range(10)), "y": (i + 10 for i in range(10))}, (0.6, 0.4))
        self.assertEqual(sorted(parts[0]["x"] + parts[1]["x"]), list(range(10)))
        for part in parts:
            self.assertEqual(part["y"], [i + 10 for i in part["x"]])

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
    def test_one_shot_inputs_are_rejected_without_consumption(self):
        validator = TypedDictIterable({"x": int})
        for make_iterator in (lambda: (i for i in [1, 2]), lambda: iter([1, 2]), SizedOneShotIterator):
            for method in (validator, lambda data: validator.check_type("x", data["x"])):
                value = make_iterator()
                with self.subTest(iterator=type(value)), self.assertRaisesRegex(TypeError, "one-shot"):
                    method({"x": value})
                if isinstance(value, SizedOneShotIterator):
                    self.assertEqual(value.reads, 0)
                self.assertEqual(list(value), [1, 2])

    def test_file_iterator_is_rejected_without_consumption(self):
        value = StringIO("first\nsecond\n")
        with self.assertRaisesRegex(TypeError, "one-shot"):
            TypedDictIterable({"x": str})({"x": value})
        self.assertEqual(value.tell(), 0)

    def test_unsized_reiterable_is_rejected_clearly(self):
        value = UnsizedIterable()
        with self.assertRaisesRegex(TypeError, "sized"):
            TypedDictIterable({"x": int})({"x": value})
        self.assertEqual(value.iterations, 0)

    def test_sized_collections_are_preserved_and_can_be_validated_again(self):
        validator = TypedDictIterable({"x": int, "y": int, "z": int})
        data = {"x": [1, 2], "y": (3, 4), "z": range(2)}
        for _ in range(2):
            self.assertIs(validator(data), data)
            self.assertIs(validator(data)["y"], data["y"])

    def test_empty_columns_and_empty_schema(self):
        data = {"x": [], "y": ()}
        self.assertIs(TypedDictIterable({"x": int, "y": str})(data), data)
        self.assertEqual(TypedDictIterable({})({}), {})
        with self.assertRaises(ValueError):
            TypedDictIterable({"x": int, "y": int})({"x": [], "y": [1]})
        with self.assertRaises(KeyError):
            TypedDictIterable({"x": int})({})

    def test_union_type_errors_do_not_raise_attribute_errors(self):
        validator = TypedDictIterable({"x": int | None})
        self.assertEqual(validator({"x": [1, None]}), {"x": [1, None]})
        for value in (["wrong"], "wrong", 1, None):
            with self.subTest(value=value), self.assertRaises(TypeError):
                validator({"x": value})

    def test_rejected_data_assignment_preserves_previous_data_and_iterator(self):
        validator = TypedDictIterable({"x": int})
        previous = {"x": [1, 2]}
        validator.data = previous
        values = iter([3, 4])
        with self.assertRaisesRegex(TypeError, "one-shot"):
            validator.data = {"x": values}
        self.assertIs(validator.data, previous)
        self.assertEqual(list(values), [3, 4])

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
    def test_all_tuple_length_combinations(self):
        dataset = Dataset.from_dict({"data": [1]})
        for sizes in product(range(4), repeat=3):
            with self.subTest(sizes=sizes):
                datasets, flags, names = sizes
                config = ValidationDatasetsDict((dataset,) * datasets, (False,) * flags,
                                                tuple("valid-{}".format(i) for i in range(names)))
                self.assertEqual(config.is_valid(), datasets == flags == names and datasets > 0)

    def test_duplicate_names_are_rejected_including_demo_sets(self):
        dataset = Dataset.from_dict({"data": [1]})
        config = ValidationDatasetsDict((dataset, dataset), (False, True), ("valid", "valid"))
        self.assertFalse(config.is_valid())

    def test_empty_datasets_are_rejected(self):
        empty = Dataset.from_dict({"data": []})
        self.assertFalse(ValidationDatasetsDict((empty,), (False,), ("valid",)).is_valid())
        self.assertFalse(ValidationDatasetsDict((empty,), (True,), ("demo",)).is_valid())

    def test_same_dataset_can_have_distinct_validation_names(self):
        dataset = Dataset.from_dict({"data": [1]})
        self.assertTrue(ValidationDatasetsDict((dataset, dataset), (False, True), ("valid", "demo")).is_valid())

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
