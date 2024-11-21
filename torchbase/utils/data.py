from datasets import Dataset

from typing import Tuple, Iterable, List, Any, Dict, Type, get_args
import types
from dataclasses import dataclass

import random


def random_split_iterable(iterable_input: Iterable[Any], portions: Tuple[float, ...]) -> Tuple[List[Any], ...]:
    if not isinstance(iterable_input, Iterable) or isinstance(iterable_input, str):
        raise TypeError("`iterable_input` must be an iterable.")

    if not isinstance(portions, tuple) or len(portions) == 0:
        raise TypeError("`portions` should be a non-empty tuple.")
    if not all(isinstance(_p, float) for _p in portions):
        raise TypeError("All portions should be floats.")
    if any(p < 0 for p in portions):
        raise ValueError("Portions should not contain negative values.")

    sum_portions = sum(portions)
    if sum_portions == 0:
        raise ValueError("Sum of portions must be greater than 0.")

    portions = [_p / sum_portions for _p in portions]

    if hasattr(iterable_input, 'read'):
        list_input = [line.strip() for line in iterable_input]
    else:
        list_input = list(iterable_input)

    random.shuffle(list_input)

    num_all = len(list_input)
    split_indices = [int(sum(portions[:i + 1]) * num_all) for i in range(len(portions))]

    splits = []
    start_idx = 0
    for end_idx in split_indices:
        splits.append(list_input[start_idx:end_idx])
        start_idx = end_idx

    return tuple(splits)


def split_to_train_valid_test(iterable_input: Iterable, portions: Tuple[float, float, float] = (0.75, 0.15, 0.15)) -> \
        Tuple[
            List, List, List]:
    if len(portions) != 3:
        raise ValueError("`portions` should be a tuple of size 3 specifying train-valid-test portions, respectively.")
    train, valid, test = random_split_iterable(iterable_input, portions=portions)

    return train, valid, test


class TypedDict:
    def __init__(self, type_dict: Dict[str, Type]):
        if not isinstance(type_dict, Dict):
            raise TypeError("type_dict should be a dictionary.")
        for key, value in type_dict.items():
            if not isinstance(key, str):
                raise TypeError("key `{}` of type_dict should be a string.".format(key))
            if not isinstance(value, type) and not isinstance(value, types.UnionType):
                raise TypeError("Specify the expected type for the key `{}`. Union of types allowed".format(key))

        self.type_dict = type_dict

    def check_type(self, key: str, value: Any) -> None:
        if key not in self.type_dict.keys():
            raise KeyError("`{}` not a recognized key.".format(key))

        expected_type = self.type_dict[key]
        none_allowed = type(None) in get_args(expected_type)

        if value is None and not none_allowed:
            raise TypeError("Value for `{}` is not allowed to be None.".format(key))

        if not isinstance(value, expected_type):
            raise TypeError("Got the unexpected type {} for the key {}.".format(type(value).__name__, key))

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in data_dict.items():
            self.check_type(key, value)

        for key in self.type_dict.keys():
            if key not in data_dict.keys():
                raise TypeError(
                    "The declared key `{}` in type_dict is missing in the passed data_dict.".format(key))

        return data_dict


class TypedDictIterable(TypedDict):

    def __init__(self, type_dict: Dict[str, Type]):
        super().__init__(type_dict)
        self._data = None

    def check_type(self, key: str, value: Any) -> None:
        if key not in self.type_dict.keys():
            raise KeyError("`{}` not a recognized key.".format(key))

        expected_type = self.type_dict[key]
        none_allowed = type(None) in get_args(expected_type)

        if value is None and not none_allowed:
            raise TypeError("Value for `{}` is not allowed to be None.".format(key))

        if isinstance(value, (str, bytes)):
            raise TypeError(
                "Value for `{}` is expected to be an iterable of {}, got a string/bytes instead.".format(
                    key, expected_type.__name__))
        if not isinstance(value, Iterable):
            raise TypeError(
                "Value for `{}` is expected to be an iterable of {}, got non-iterable.".format(key,
                                                                                               expected_type.__name__))
        for item in value:
            if not isinstance(item, expected_type):
                raise TypeError(
                    "Expected items in `{}` to be `{}`, got `{}` instead.".format(key, expected_type.__name__,
                                                                                  type(item).__name__))

    def __call__(self, data_dict: Dict[str, Iterable[Any]]) -> Dict[str, Iterable[Any]]:
        for key in self.type_dict.keys():
            if key not in data_dict:
                raise KeyError("The declared key `{}` in type_dict is missing in the passed data_dict.".format(key))

        last_iterable_length = -1
        for idx, (key, value) in enumerate(data_dict.items()):
            self.check_type(key, value)
            this_iterable_length = len(list(value))
            if idx != 0 and this_iterable_length != last_iterable_length:
                raise ValueError("The length of all iterables over values must be the same.")

            last_iterable_length = this_iterable_length

        return data_dict

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value: Dict[str, Any]):
        self._data = self.__call__(value)


@dataclass
class ValidationDatasetsDict:
    datasets: Tuple[Dataset, ...]
    only_for_demo: Tuple[bool, ...]
    names: Tuple[str, ...]

    def is_valid(self) -> bool:
        if not isinstance(self.datasets, tuple):
            return False
        if any(not isinstance(dataset, Dataset) for dataset in self.datasets):
            return False
        if not isinstance(self.only_for_demo, tuple):
            return False
        if any(not isinstance(question, bool) for question in self.only_for_demo):
            return False
        if not isinstance(self.names, tuple):
            return False
        if any(not isinstance(name, str) for name in self.names):
            return False
        if len(self.datasets) != len(self.only_for_demo) != len(self.names):
            return False

        return True
