from dataclasses import dataclass, fields, asdict

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
