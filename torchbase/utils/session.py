from dataclasses import dataclass, fields, asdict

from typing import Dict, Tuple

import datetime
import socket


def get_current_time_tag() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def generate_log_dir_tag(tag_postfix: None | str = None) -> str:
    current_datetime = get_current_time_tag()
    hostname = socket.gethostname()
    tag = "{}_{}".format(current_datetime, hostname)
    if tag_postfix is not None:
        if not isinstance(tag_postfix, str):
            raise TypeError("`tag_postfix`, if specified, should be a string.")
        tag += "_{}".format(tag_postfix)

    return tag


def is_custom_scalar_logging_layout_valid(layout: Dict,
                                          validation_dataset_names: Tuple[str, ...],
                                          metric_names: Tuple[str, ...]) -> bool:
    if not isinstance(layout, dict):
        return False

    def validate_subgroup(subgroup):
        if not isinstance(subgroup, dict):
            return False

        for key, value in subgroup.items():
            if not isinstance(key, str):
                return False

            if isinstance(value, list):
                if len(value) != 2:
                    return False
                if not isinstance(value[0], str) or value[0] not in ['Multiline', 'Margin', 'Line', 'Bar']:
                    return False
                if not isinstance(value[1], list):
                    return False
                for tag in value[1]:
                    if not isinstance(tag, str):
                        return False
                    if len(tag.split("/")) != 3:
                        return False
                    if tag.split("/")[0] not in ["training"] + ["validation-{}".format(name) for name in
                                                                validation_dataset_names]:
                        return False
                    if tag.split("/")[1] not in ["loss"] + list(metric_names):
                        return False
                    if tag.split("/")[2] not in ["epochs", "iterations"]:
                        return False

            elif isinstance(value, dict):
                if not validate_subgroup(value):
                    return False
            else:
                return False

        return True

    return validate_subgroup(layout)


@dataclass
class TrainingConfigSessionDict:
    device_name: str
    num_epochs: int
    mini_batch_size: int
    learning_rate: float
    weight_decay: float = 0.0
    dataloader_num_workers: int = 0
    loss_function_params: None | dict = None

    def __init__(self, config: dict):
        for field_info in fields(self):
            field_name = field_info.name
            if field_name not in config.keys():
                config[field_name] = field_info.default

        for key, value in config.items():
            setattr(self, key, value)
        if not self.is_valid():
            raise ValueError("The passed `config` does not match the required types. Debug to see which field(s) fail.")

    def to_dict(self) -> dict:
        # TODO: Add test
        return asdict(self)

    def is_valid(self) -> bool:
        if not isinstance(self.device_name, str):
            return False
        if not isinstance(self.num_epochs, int):
            return False
        if self.num_epochs <= 0:
            return False
        if not isinstance(self.mini_batch_size, int):
            return False
        if self.mini_batch_size <= 0:
            return False
        if not isinstance(self.learning_rate, float):
            return False
        if self.learning_rate <= 0:
            return False
        if not isinstance(self.weight_decay, float):
            return False
        if self.weight_decay < 0:
            return False
        if not isinstance(self.dataloader_num_workers, int):
            return False
        if self.dataloader_num_workers < 0:
            return False
        if self.loss_function_params is not None:
            if not isinstance(self.loss_function_params, dict):
                return False

        return True
