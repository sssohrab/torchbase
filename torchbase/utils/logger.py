from dataclasses import dataclass, fields, asdict

from typing import List, Dict

import json


@dataclass
class ProgressManager:
    iter_current_epoch: int = 0
    iter_total: int = 0
    samples_current_iter: int = 0
    samples_current_epoch: int = 0
    samples_total: int = 0
    epoch: int = 0

    def increment_iter(self, mini_batch_size: int) -> None:
        if not isinstance(mini_batch_size, int) or mini_batch_size < 0:
            raise ValueError("`mini_batch_size` should be a positive integer.")

        self.samples_current_iter = mini_batch_size
        self.iter_current_epoch += 1
        self.iter_total += 1
        self.samples_current_epoch += mini_batch_size
        self.samples_total += mini_batch_size

    def reset_epoch(self) -> None:
        self.samples_current_iter = 0
        self.iter_current_epoch = 0
        self.samples_current_epoch = 0

    def increment_epoch(self) -> None:
        self.epoch += 1
        self.reset_epoch()

    def reset(self) -> None:
        self.iter_current_epoch = 0
        self.iter_total = 0
        self.samples_current_iter = 0
        self.samples_current_epoch = 0
        self.samples_total = 0
        self.epoch = 0

    def serialize_to_disk(self, path: str) -> None:
        with open(path, "w") as file:
            json.dump(asdict(self), file, indent=2)

    def set_fields_from_disk(self, path: str) -> None:
        with open(path, 'r') as file:
            dict_data = json.load(file)

        for field in fields(self):
            setattr(self, field.name, dict_data[field.name])


class ValuesLogger:
    def __init__(self, names: List[str], progress_manager: ProgressManager) -> None:
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise TypeError("Provide `names` as a list of strings to specify all the values you want to log.")
        self.names = names

        if not isinstance(progress_manager, ProgressManager):
            raise TypeError("Pass a `progress_manager` from `ProgressManager` class.")
        self.progress_manager = progress_manager

        self.current_values: Dict[str, float] = {name: 0.0 for name in names}
        self.average_of_epoch: Dict[str, float] = {name: 0.0 for name in names}
        self.average_overall: Dict[str, float] = {name: 0.0 for name in names}

    def update(self, values_dict: Dict[str, float]) -> None:
        if not isinstance(values_dict, dict) or any(
                [not isinstance(name, str) or not isinstance(value, float) for name, value in values_dict.items()]):
            raise TypeError("The passed `values_dict` must be a dictionary of string-float pairs.")
        if set(values_dict.keys()) != set(self.names):
            raise ValueError("The passed `values_dict` does not have the same keys as the declared `self.names`.")

        self.current_values = values_dict
        current_samples = self.progress_manager.samples_current_iter
        total_samples = self.progress_manager.samples_total

        for name, value in values_dict.items():
            old_avg_epoch = self.average_of_epoch[name]
            new_avg_epoch = old_avg_epoch + (
                    value - old_avg_epoch) * current_samples / self.progress_manager.samples_current_epoch
            self.average_of_epoch[name] = new_avg_epoch

            old_avg_overall = self.average_overall[name]
            new_avg_overall = old_avg_overall + (value - old_avg_overall) * current_samples / total_samples
            self.average_overall[name] = new_avg_overall

    def reset_epoch(self) -> None:
        self.current_values = {name: 0.0 for name in self.names}
        self.average_of_epoch = {name: 0.0 for name in self.names}

    def reset(self) -> None:
        self.current_values = {name: 0.0 for name in self.names}
        self.average_of_epoch = {name: 0.0 for name in self.names}
        self.average_overall = {name: 0.0 for name in self.names}
        self.progress_manager.reset()

    def serialize_to_disk(self, path: str) -> None:
        with open(path, "w") as file:
            json.dump({
                "names": self.names,
                "current_values": self.current_values,
                "average_of_epoch": self.average_of_epoch,
                "average_overall": self.average_overall
            }, file, indent=2)

    def set_state_values_from_disk(self, path: str) -> None:
        with open(path, 'r') as file:
            dict_data = json.load(file)
        if dict_data["names"] != self.names:
            raise RuntimeError("Inconsistent states dict loaded from disk, since `names` do not match.")

        self.current_values = dict_data["current_values"]
        self.average_of_epoch = dict_data["average_of_epoch"]
        self.average_overall = dict_data["average_overall"]
