from torchbase.utils.session import generate_log_dir_tag, TrainingConfigSessionDict
from torchbase.utils.data import ValidationDatasetsDict
from torchbase.utils.networks import load_network_from_state_dict_to_device
from torchbase.utils.logger import ProgressManager, ValuesLogger, LoggableParams

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from datasets import Dataset

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Callable
import inspect
import os
import random
import json

SAVED_NETWORK_NAME = "network.pth"
SAVED_OPTIMIZER_NAME = "optimizer.pth"
SAVED_RNG_NAME = "rng_states.pth"
ITERATION_STEPS_TO_SAVE_STATES = 10


class TrainingBaseSession(ABC):
    def __init__(self, config: Dict,
                 runs_parent_dir: None | str = None,
                 create_run_dir_afresh: bool = True,
                 source_run_dir_tag: None | str = None,
                 tag_postfix: None | str = None):

        self.config_session, self.config_data, self.config_metrics, self.config_network = self.setup_configs(config)
        self.run_dir = self.setup_run_dir_for_logging(runs_parent_dir,
                                                      create_run_dir_afresh,
                                                      source_run_dir_tag,
                                                      tag_postfix)

        self.configure_states_dir_and_randomness_sources(self.run_dir, create_run_dir_afresh)
        self.save_config_to_run_dir(create_run_dir_afresh)

        self.device = torch.device(self.config_session.device_name)  # TODO

        self.dataset_train, self.datasets_valid_dict = self._init_datasets()
        self.dataloader_train, self.dataloader_valid_dict = self.init_dataloaders()

        self.network = self._init_network()

        self.optimizer = self.init_optimizer()

        self.load_network_and_optimizer_states_if_relevant(source_run_dir_tag, create_run_dir_afresh)

        self.writer = SummaryWriter(log_dir=self.run_dir)

        self.progress_train = ProgressManager()
        self.loggable_train = LoggableParams({"loss": self.get_loss_value})
        self.value_logger_train = ValuesLogger(self.loggable_train.get_names(), progress_manager=self.progress_train)

        self.progress_valid_dict = {valid_dataset_name: ProgressManager() for valid_dataset_name in
                                    self.datasets_valid_dict.names}
        self.loggable_valid_dict = {
            valid_dataset_name: LoggableParams({"loss": self.get_loss_value})
            for valid_dataset_name in self.datasets_valid_dict.names}
        self.value_logger_valid_dict = {
            valid_dataset_name: ValuesLogger(self.loggable_train.get_names(),
                                             progress_manager=self.progress_valid_dict[valid_dataset_name]) for
            valid_dataset_name in self.datasets_valid_dict.names}

        self.best_validation_loss_dict: Dict[str, Tuple[float, int]] = {}

    @staticmethod
    def print(report: str, end: str | None = None) -> None:
        # TODO (#13): Maybe using the `logging` package to handle various levels of messages.
        # TODO (#13): A global mechanism to activate/deactivate printing, writing to log files, etc.
        print(report, end=end)

    @staticmethod
    def setup_configs(config: Dict) -> Tuple[TrainingConfigSessionDict, Dict, Dict, Dict]:
        if not isinstance(config, dict):
            raise TypeError("Pass a python dictionary as session `config`.")
        expected_keys = ["session", "data", "network", "metrics"]
        if not all(key in config.keys() for key in expected_keys):
            raise ValueError("`config` should specify all of these fields {}".format(expected_keys))

        if "architecture" not in config["network"].keys():
            raise ValueError(
                "The 'network' `config` should specify the class name of the network's 'architecture' as a field.")

        return TrainingConfigSessionDict(config["session"]), config["data"], config["metrics"], config["network"]

    @staticmethod
    def setup_run_dir_for_logging(runs_parent_dir: None | str = None,
                                  create_run_dir_afresh: bool = True,
                                  source_run_dir_tag: None | str = None,
                                  tag_postfix: None | str = None) -> str:

        if runs_parent_dir is None:
            runs_parent_dir = os.path.join(os.getcwd(), "runs")
        else:
            if not isinstance(runs_parent_dir, str):
                raise TypeError("You specified `runs_parent_dir` not to be under the default current-working-dir, "
                                "but the passed value is not of type string.")
            else:
                os.makedirs(runs_parent_dir, exist_ok=True)

        if create_run_dir_afresh:
            if tag_postfix is not None:
                if not isinstance(tag_postfix, str):
                    raise TypeError("You specified to post-fix the generated run-tag, "
                                    "but the passed value is not of type string.")

            run_dir_tag = generate_log_dir_tag(tag_postfix)
            os.makedirs(os.path.join(runs_parent_dir, run_dir_tag), exist_ok=False)

            return os.path.join(runs_parent_dir, run_dir_tag)

        else:
            if not isinstance(source_run_dir_tag, str):
                raise TypeError("By choosing `create_run_dir_afresh = False`, you requested to take over from an "
                                "existing `run_dir`. However, the passed `source_run_dir_tag` is not a string.")

            if not os.path.exists(os.path.join(runs_parent_dir, source_run_dir_tag)):
                raise ValueError("The source directory for logging `source_run_dir_tag = {}` does not exist "
                                 "under `runs_parent_dir = {}` ".format(source_run_dir_tag, runs_parent_dir))

            run_dir_tag = source_run_dir_tag

            return os.path.join(runs_parent_dir, run_dir_tag)

    @staticmethod
    def check_source_states_dir_is_valid(source_states_dir: str):
        if not os.path.exists(source_states_dir):
            raise FileNotFoundError(
                "Requested to restart from the existing previous run `{}`, "
                "but its states_dir to reload from is missing.".format(os.path.split(source_states_dir)[-2]))
        if not os.path.exists(os.path.join(source_states_dir, SAVED_RNG_NAME)):
            raise FileNotFoundError("The randomness states file does not exist under the source `{}`. "
                                    "This is not a valid source".format(source_states_dir))

        if not os.path.exists(os.path.join(source_states_dir, SAVED_NETWORK_NAME)):
            raise FileNotFoundError("The saved network's `states_dict` does not exist under the source `{}`. "
                                    "This is not a valid source".format(source_states_dir))

        if not os.path.exists(os.path.join(source_states_dir, SAVED_OPTIMIZER_NAME)):
            raise FileNotFoundError("The optimizer `states_dict` file does not exist in the source `{}`. "
                                    "This is not a valid source".format(source_states_dir))

    def init_best_validation_loss_dict(self) -> Dict[str, Tuple[float, int]]:
        best_validation_loss_dict = {
            self.datasets_valid_dict.names[ind]: (float("inf"), -2) for
            ind in range(len(self.datasets_valid_dict.names)) if not self.datasets_valid_dict.only_for_demo[ind]}

        return best_validation_loss_dict

    def configure_states_dir_and_randomness_sources(self, run_dir: str, create_run_dir_afresh: bool) -> None:
        states_dir = os.path.join(run_dir, "states")

        if create_run_dir_afresh:
            os.makedirs(states_dir, exist_ok=False)
            torch.save({"torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                        "numpy_rng_state": np.random.get_state(),
                        "random_rng_state": random.getstate()}, os.path.join(states_dir, SAVED_RNG_NAME))

            self.best_validation_loss_dict = self.init_best_validation_loss_dict()

        else:
            TrainingBaseSession.check_source_states_dir_is_valid(states_dir)
            # TODO (#9): The warning regarding weight_only option. Instead of pickling, just serialize them with json.
            rng_states = torch.load(os.path.join(states_dir, SAVED_RNG_NAME))
            torch.set_rng_state(rng_states["torch_rng_state"])
            if rng_states["cuda_rng_state"] is not None:
                torch.cuda.set_rng_state(rng_states["cuda_rng_state"])
            np.random.set_state(rng_states["numpy_rng_state"])
            random.setstate(rng_states["random_rng_state"])

            with open(os.path.join(states_dir, "best_validation_loss_dict.json"), "r") as f:
                self.best_validation_loss_dict = json.load(f)

    def save_config_to_run_dir(self, create_run_dir_afresh: bool) -> None:
        config = {
            "session": self.config_session.to_dict(),
            "data": self.config_data,
            "metrics": self.config_metrics,
            "network": self.config_network
        }
        if create_run_dir_afresh:
            _postfix = ""
        else:
            _postfix = "_{}".format(generate_log_dir_tag(None))

        with open(os.path.join(self.run_dir, "config{}.json".format(_postfix)), "w") as file:
            json.dump(config, file, indent=2)

    @abstractmethod
    def init_datasets(self) -> Tuple[Dataset, ValidationDatasetsDict]:
        pass

    def _init_datasets(self) -> Tuple[Dataset, ValidationDatasetsDict]:

        dataset_train, datasets_valid_dict = self.init_datasets()

        if not isinstance(dataset_train, Dataset):
            raise ValueError("`dataset_train` should be an instance of `datasets.Dataset`.")
        if not isinstance(datasets_valid_dict, ValidationDatasetsDict):
            raise ValueError("`datasets_valid_dict` must be an instance of `ValidationDatasetsDict`.")
        if not datasets_valid_dict.is_valid():
            raise ValueError("Failed to create a valid `datasets_valid_dict`, an instance of `ValidationDatasetsDict`.")

        dataset_train.set_format("torch")
        for dataset in datasets_valid_dict.datasets:
            dataset.set_format("torch")

        return dataset_train, datasets_valid_dict

    def init_dataloaders(self) -> Tuple[DataLoader, Dict[str, DataLoader]]:
        # TODO (#13): `datasets.Dataset.shuffle()` or `torch.utils.data.DataLoader` ?
        # TODO (#13): How about `torchdata.StatefulDataLoader` for full reproducibility of checkpoints?
        # TODO (#10): When streaming mode gets support, dataloading and shuffling will get nuanced.

        dataloader_train = DataLoader(self.dataset_train, batch_size=self.config_session.mini_batch_size, shuffle=True,
                                      num_workers=self.config_session.dataloader_num_workers,
                                      collate_fn=self.dataloader_collate_function)
        dataloader_valid_dict = {
            self.datasets_valid_dict.names[ind]: DataLoader(self.datasets_valid_dict.datasets[ind],
                                                            batch_size=self.config_session.mini_batch_size,
                                                            shuffle=True,
                                                            num_workers=self.config_session.dataloader_num_workers,
                                                            collate_fn=self.dataloader_collate_function) for ind in
            range(len(self.datasets_valid_dict.names))}

        return dataloader_train, dataloader_valid_dict

    def dataloader_collate_function(self, batch: List[Any]) -> Dict[str, List[Any] | torch.Tensor]:
        return torch.utils.data.default_collate(batch)

    @abstractmethod
    def init_network(self) -> torch.nn.Module:
        pass

    def _init_network(self) -> torch.nn.Module:

        network = self.init_network()
        if not isinstance(network, torch.nn.Module):
            raise TypeError("Failed to instantiate a valid `network`, an instance of `torch.nn.Module`.")

        expected_network_class_name = self.config_network["architecture"]
        if network.__class__.__name__ != expected_network_class_name:
            raise TypeError(
                "The loaded network is an instance of `{}`, whereas the network config was assuming"
                " an instance of `{}` to be instantiated. Check the implementation of the abstract method"
                "`init_network()` for errors or modify your network config accordingly.".format(
                    network.__class__.__name__, expected_network_class_name))

        return network

    def init_optimizer(self) -> torch.optim:
        optimizer = torch.optim.Adam(self.network.parameters(),
                                     lr=self.config_session.learning_rate,
                                     weight_decay=self.config_session.weight_decay)

        return optimizer

    def save_network_and_optimizer_states(self) -> None:
        torch.save(self.network.state_dict(), os.path.join(self.run_dir, "states", SAVED_NETWORK_NAME))
        torch.save(self.optimizer.state_dict(), os.path.join(self.run_dir, "states", SAVED_OPTIMIZER_NAME))

    def load_network_and_optimizer_states_if_relevant(self, source_run_dir_tag: None | str,
                                                      create_run_dir_afresh: bool) -> None:
        if source_run_dir_tag is None:
            if not create_run_dir_afresh:
                raise ValueError("This cannot happen anyway.")
            return

        source_states_dir_path = os.path.join(os.path.dirname(self.run_dir), source_run_dir_tag, "states")
        self.network = load_network_from_state_dict_to_device(
            self.network,
            state_dict_path=os.path.join(source_states_dir_path, SAVED_NETWORK_NAME),
            device=self.device)

        if create_run_dir_afresh:
            return
        # Optimizer states loaded only for fresh run, but network states loaded anyway (if source run is specified).
        self.optimizer.load_state_dict(
            torch.load(os.path.join(source_states_dir_path, SAVED_OPTIMIZER_NAME), weights_only=True))

    def save_training_states(self) -> None:
        states_dir_path = os.path.join(self.run_dir, "states")
        torch.save(self.network.state_dict(), os.path.join(states_dir_path, SAVED_NETWORK_NAME))
        torch.save(self.optimizer.state_dict(), os.path.join(states_dir_path, SAVED_OPTIMIZER_NAME))

        # TODO: learning rate scheduler state_dicts
        # TODO (#13): Dataloader states dict (maybe from torchdata.StatefulDataLoader)

        self.progress_train.serialize_to_disk(os.path.join(states_dir_path, "progress_manager_train.json"))
        self.value_logger_train.serialize_to_disk(os.path.join(states_dir_path, "values_logger_train.json"))

    def save_progress_and_log_states_for_valid_set(self, valid_dataset_name: str):
        states_dir_path = os.path.join(self.run_dir, "states")
        self.progress_valid_dict[valid_dataset_name].serialize_to_disk(
            os.path.join(states_dir_path, "progress_manager_valid_{}.json".format(valid_dataset_name)))
        self.value_logger_valid_dict[valid_dataset_name].serialize_to_disk(
            os.path.join(states_dir_path, "values_logger_valid_{}.json".format(valid_dataset_name)))

        with open(os.path.join(states_dir_path, "best_validation_loss_dict.json"), "w") as f:
            json.dump(self.best_validation_loss_dict, f, indent=2)

    @abstractmethod
    def forward_pass(self, mini_batch: Dict[str, Any | torch.Tensor]) -> Dict[str, Any | torch.Tensor]:
        inp = mini_batch["some_key"].to(self.device)
        out = self.network(inp)

        _dict = {"input": inp,
                 "output": out}  # Keys corresponding to anything the loss function or metrics calculation would need.

        return _dict

    @abstractmethod
    def loss_function(self, **kwargs: Any) -> torch.Tensor:
        # Implement a keyword-only function with keys included in `self.forward_pass` output dictionary.
        # Optionally use some params from  self.config_session.loss_function_params

        return torch.empty(requires_grad=True)

    @staticmethod
    def get_loss_value(*, loss_tensor: torch.Tensor) -> float:
        # Override if `self.loss_function` provides non-scalar outputs.
        return loss_tensor.item()

    @staticmethod
    def infer_mini_batch_size(mini_batch: Dict[str, Any | torch.Tensor]) -> int:
        mini_batch_size = None
        for key, val in mini_batch.items():
            if isinstance(val, torch.Tensor):
                if mini_batch_size is None:
                    mini_batch_size = val.shape[0]  # Assuming PyTorch convention.
                else:
                    this_mini_batch_size = val.shape[0]
                    if this_mini_batch_size != mini_batch_size:
                        raise RuntimeError("Inconsistent sizes between different fields of the mini-batch.")

        if mini_batch_size is None:
            raise RuntimeError("Did not manage to infer the mini-batch size from the provided `mini_batch`. Check"
                               "the implementation of `self.dataloader_collate_function`")

        return mini_batch_size

    def do_one_training_iteration(self, mini_batch: Dict[str, Any | torch.Tensor]) -> None:
        self.network.train()
        outs_dict = self.forward_pass(mini_batch)
        loss_function_signature = inspect.signature(self.loss_function)
        loss = self.loss_function(**{k: v for k, v in outs_dict.items() if k in loss_function_signature.parameters})
        assert not torch.isnan(loss), "A NaN value detected during loss function evaluation."

        self.optimizer.zero_grad()
        loss.backward()
        # TODO: torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.progress_train.increment_iter(self.infer_mini_batch_size(mini_batch))
        # `outs_dict` is supposed to have all key-value pairs required by functionals in metrics.
        self.value_logger_train.update(self.loggable_train(**{"loss_tensor": loss}, **outs_dict))

        for param in self.value_logger_train.names:
            self.writer.add_scalar("training/{}/iterations".format(param),
                                   self.value_logger_train.current_values[param],
                                   self.progress_train.iter_total)

    def do_one_validation_iteration(self, mini_batch: Dict[str, Any | torch.Tensor], valid_dataset_name: str) -> None:
        self.network.eval()
        with torch.no_grad():
            outs_dict = self.forward_pass(mini_batch)

        loss_function_signature = inspect.signature(self.loss_function)
        loss = self.loss_function(**{k: v for k, v in outs_dict.items() if k in loss_function_signature.parameters})
        self.progress_valid_dict[valid_dataset_name].increment_iter(self.infer_mini_batch_size(mini_batch))
        self.value_logger_valid_dict[valid_dataset_name].update(
            self.loggable_valid_dict[valid_dataset_name](**{"loss_tensor": loss}, **outs_dict))

        for param in self.value_logger_valid_dict[valid_dataset_name].names:
            self.writer.add_scalar("validation-{}/{}/iterations".format(valid_dataset_name, param),
                                   self.value_logger_valid_dict[valid_dataset_name].current_values[param],
                                   self.progress_valid_dict[valid_dataset_name].iter_total)
