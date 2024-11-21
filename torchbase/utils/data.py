import random
from typing import Tuple, Iterable, List, Any


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
