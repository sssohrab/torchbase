from dataclasses import dataclass, fields, asdict

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
        if self.weight_decay <= 0:
            return False
        if not isinstance(self.dataloader_num_workers, int):
            return False
        if self.dataloader_num_workers < 0:
            return False
        if self.loss_function_params is not None:
            if not isinstance(self.loss_function_params, dict):
                return False

        return True
