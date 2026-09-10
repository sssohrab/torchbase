from dataclasses import dataclass, fields, field, asdict, MISSING
from copy import deepcopy
import math
from typing import Dict, Tuple

import torch
import numpy as np
import random

import json
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
                    if tag.split("/")[2] not in ["epochs", "iterations", "batch_means"]:
                        return False

            elif isinstance(value, dict):
                if not validate_subgroup(value):
                    return False
            else:
                return False

        return True

    return validate_subgroup(layout)


def _copy_config(config: dict) -> dict:
    """Copy JSON-compatible settings, allowing tuples as JSON arrays."""
    try:
        json.dumps(config, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("The config must contain JSON-compatible values: {}".format(error)) from error

    def check_keys(value):
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("All config dictionary keys must be strings, including nested settings.")
            for item in value.values():
                check_keys(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                check_keys(item)

    check_keys(config)
    return deepcopy(config)


@dataclass
class TrainingConfigSessionDict:
    """Declared session settings, copied from the caller; unknown fields are rejected."""
    device_name: str
    num_epochs: int
    mini_batch_size: int
    learning_rate: float
    weight_decay: float = 0.0
    dataloader_num_workers: int = 0
    checkpoint_interval: int = 100
    loss_function_params: None | dict = None

    def __init__(self, config: dict):
        if not isinstance(config, dict):
            raise TypeError("The session config must be a dictionary.")
        expected = {field_info.name for field_info in fields(self)}
        unknown = config.keys() - expected
        if unknown:
            raise ValueError("Unknown session config fields: {}".format(
                ", ".join(sorted(map(str, unknown)))))
        missing = [info.name for info in fields(self) if info.default is MISSING and info.name not in config]
        if missing:
            raise ValueError("Missing required session config fields: {}".format(", ".join(missing)))
        for field_info in fields(self):
            setattr(self, field_info.name, config.get(field_info.name, field_info.default))
        invalid = self._invalid_fields()
        if invalid:
            raise ValueError("Invalid session config fields: {}".format(", ".join(invalid)))
        if self.loss_function_params is not None:
            self.loss_function_params = _copy_config(self.loss_function_params)

    def to_dict(self) -> dict:
        return asdict(self)

    def is_valid(self) -> bool:
        return not self._invalid_fields()

    def _invalid_fields(self) -> list[str]:
        checks = {
            "device_name": isinstance(self.device_name, str) and bool(self.device_name),
            "num_epochs": type(self.num_epochs) is int and self.num_epochs > 0,
            "mini_batch_size": type(self.mini_batch_size) is int and self.mini_batch_size > 0,
            "learning_rate": isinstance(self.learning_rate, float) and math.isfinite(self.learning_rate)
                             and self.learning_rate > 0,
            "weight_decay": isinstance(self.weight_decay, float) and math.isfinite(self.weight_decay)
                            and self.weight_decay >= 0,
            "dataloader_num_workers": type(self.dataloader_num_workers) is int and self.dataloader_num_workers >= 0,
            "checkpoint_interval": type(self.checkpoint_interval) is int and self.checkpoint_interval > 0,
            "loss_function_params": self.loss_function_params is None or isinstance(self.loss_function_params, dict),
        }
        return [name for name, valid in checks.items() if not valid]


@dataclass
class RandomnessGeneratorStates:
    torch_state: bytes = field(default_factory=lambda: torch.get_rng_state().numpy().tobytes())
    cuda_state: bytes = field(
        default_factory=lambda: torch.cuda.get_rng_state().numpy().tobytes() if torch.cuda.is_available() else b"")
    numpy_state: tuple = field(default_factory=lambda: np.random.get_state())
    random_state: tuple = field(default_factory=lambda: random.getstate())

    def to_dict(self):
        return {
            "torch_state": self.torch_state.hex(),
            "cuda_state": self.cuda_state.hex() if self.cuda_state else "",
            "numpy_state": (
                self.numpy_state[0],
                self.numpy_state[1].tolist(),
                *self.numpy_state[2:]
            ),
            "random_state": (self.random_state[0], list(self.random_state[1]), self.random_state[2])
        }

    def save(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filename: str):
        with open(filename, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data):
        rng_state = cls()
        rng_state.torch_state = bytes.fromhex(data["torch_state"])
        rng_state.cuda_state = bytes.fromhex(data["cuda_state"])

        numpy_state = (
            data["numpy_state"][0],
            np.array(data["numpy_state"][1], dtype=np.uint32),
            *data["numpy_state"][2:]
        )
        rng_state.numpy_state = numpy_state

        rng_state.random_state = (
            data["random_state"][0],
            tuple(data["random_state"][1]),
            data["random_state"][2]
        )

        return rng_state

    def apply(self):
        torch.set_rng_state(torch.tensor(np.frombuffer(self.torch_state, dtype=np.uint8)))
        if self.cuda_state and torch.cuda.is_available():
            torch.cuda.set_rng_state(torch.tensor(np.frombuffer(self.cuda_state, dtype=np.uint8)))

        numpy_state = (
            str(self.numpy_state[0]),
            np.array(self.numpy_state[1], dtype=np.uint32),
            int(self.numpy_state[2]),
            int(self.numpy_state[3]),
            float(self.numpy_state[4])
        )
        np.random.set_state(numpy_state)

        random.setstate(self.random_state)
