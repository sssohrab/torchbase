from dataclasses import dataclass, fields, asdict

from typing import List, Dict, Any, Callable

import inspect
import json
from copy import deepcopy

from torchbase.utils.metrics import EpochMetric


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
            json.dump(self.state_dict(), file, indent=2)

    def state_dict(self) -> Dict:
        return asdict(self)

    def load_state_dict(self, dict_data: Dict) -> None:
        for field in fields(self):
            setattr(self, field.name, dict_data[field.name])

    def set_fields_from_disk(self, path: str) -> None:
        with open(path, 'r') as file:
            dict_data = json.load(file)

        self.load_state_dict(dict_data)


class ValuesLogger:
    """Batch-score averages and, separately, optional whole-epoch accumulators."""

    def __init__(self, names: List[str], progress_manager: ProgressManager,
                 epoch_metrics: Dict[str, EpochMetric] | None = None) -> None:
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise TypeError("Provide `names` as a list of strings to specify all the values you want to log.")
        self.names = names

        if not isinstance(progress_manager, ProgressManager):
            raise TypeError("Pass a `progress_manager` from `ProgressManager` class.")
        self.progress_manager = progress_manager
        self.epoch_metrics = dict(epoch_metrics) if epoch_metrics is not None else {}
        if any(name not in names or not isinstance(metric, EpochMetric) for name, metric in self.epoch_metrics.items()):
            raise ValueError("Epoch metrics must be EpochMetric instances with names present in the logger.")

        self.current_values: Dict[str, float] = {name: 0.0 for name in names}
        self.average_of_epoch: Dict[str, float] = {name: 0.0 for name in names}
        self.average_overall: Dict[str, float] = {name: 0.0 for name in names}

    def update(self, values_dict: Dict[str, float], *, metric_inputs: Dict | None = None) -> None:
        if not isinstance(values_dict, dict) or any(
                [not isinstance(name, str) or not isinstance(value, float) for name, value in values_dict.items()]):
            raise TypeError("The passed `values_dict` must be a dictionary of string-float pairs.")
        if set(values_dict.keys()) != set(self.names):
            raise ValueError("The passed `values_dict` does not have the same keys as the declared `self.names`.")

        for metric in self.epoch_metrics.values():
            if metric_inputs is None:
                raise ValueError("Epoch metrics require metric_inputs when updating the logger.")
            signature = inspect.signature(metric.update)
            metric.update(**{key: value for key, value in metric_inputs.items() if key in signature.parameters})

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
        for metric in self.epoch_metrics.values():
            metric.reset()

    @property
    def epoch_values(self) -> Dict[str, float]:
        """Whole-epoch results only; average_of_epoch remains an average of batch scores."""
        values = {name: metric.compute() for name, metric in self.epoch_metrics.items()}
        if any(not isinstance(value, float) for value in values.values()):
            raise TypeError("EpochMetric.compute must return a float.")
        return values

    def reset(self) -> None:
        self.reset_epoch()
        self.average_overall = {name: 0.0 for name in self.names}
        self.progress_manager.reset()

    def serialize_to_disk(self, path: str) -> None:
        with open(path, "w") as file:
            json.dump(self.state_dict(), file, indent=2)

    def state_dict(self) -> Dict:
        return {
            "names": self.names,
            "current_values": self.current_values,
            "average_of_epoch": self.average_of_epoch,
            "average_overall": self.average_overall,
            "epoch_metrics": {name: {"type": type(metric).__module__ + "." + type(metric).__qualname__,
                                      "state": deepcopy(metric.state_dict())}
                              for name, metric in self.epoch_metrics.items()},
        }

    def set_state_values_from_disk(self, path: str) -> None:
        with open(path, 'r') as file:
            dict_data = json.load(file)
        self.load_state_dict(dict_data)

    def load_state_dict(self, dict_data: Dict) -> None:
        if dict_data["names"] != self.names:
            raise RuntimeError("Inconsistent states dict loaded from disk, since `names` do not match.")
        states = dict_data.get("epoch_metrics")
        if not isinstance(states, dict) or set(states) != set(self.epoch_metrics):
            raise RuntimeError("Checkpoint epoch metrics do not match the current logger.")
        for name, metric in self.epoch_metrics.items():
            if states[name]["type"] != type(metric).__module__ + "." + type(metric).__qualname__:
                raise RuntimeError("Checkpoint epoch metric types do not match the current logger.")
            metric.load_state_dict(deepcopy(states[name]["state"]))

        self.current_values = dict_data["current_values"]
        self.average_of_epoch = dict_data["average_of_epoch"]
        self.average_overall = dict_data["average_overall"]


class LoggableParams:
    def __init__(self, functional_dict: Dict[str, Callable[..., Any]]):
        if not isinstance(functional_dict, dict):
            raise TypeError("The passed `functional_dict` should be a dictionary with string keys and callable values.")

        for name, functional in functional_dict.items():
            self.add_functional(name, functional)

    def add_functional(self, name: str, functional: Callable) -> None:
        if not isinstance(name, str):
            raise TypeError("The `associated` to the functional should be a string.")
        if not callable(functional):
            raise TypeError("The `functional` should be a callable.")
        sig = inspect.signature(functional)

        for param in sig.parameters.values():
            if param.kind not in [inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD]:
                raise TypeError(f"Functional `{name}` must have keyword-only arguments.")

        if hasattr(self, name):
            raise AttributeError("The attribute `{}` already exists.".format(name))
        setattr(self, name, functional)

    def get_functional_dict(self) -> Dict[str, Callable[..., Any]]:
        return {name: functional for name, functional in self.__dict__.items() if callable(functional)}

    def get_names(self) -> List[str]:
        return list(self.get_functional_dict().keys())

    def evaluate_functionals(self, **kwargs) -> Dict[str, float]:
        values_dict = {}
        for name, functional in self.get_functional_dict().items():
            sig = inspect.signature(functional)
            func_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            result = functional(**func_kwargs)
            if not isinstance(result, float):
                raise ValueError("Expected the result of `{}` to be a float, got {} instead.".format(
                    name, type(result).__name__))
            values_dict[name] = result

        return values_dict

    def __call__(self, **kwargs) -> Dict[str, float]:
        return self.evaluate_functionals(**kwargs)
