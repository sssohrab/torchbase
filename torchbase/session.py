from torchbase.utils.data import ValidationDatasetsDict
from torchbase.utils.session import generate_log_dir_tag, TrainingConfigSessionDict

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

        self.device = torch.device(self.config_session.device_name)

        self.dataset_train, self.datasets_valid_dict = self._init_datasets()
        self.dataloader_train, self.dataloader_valid_dict = self.init_dataloaders()

        self.writer = SummaryWriter(log_dir=self.run_dir)

    @staticmethod
    def setup_configs(config: Dict) -> Tuple[TrainingConfigSessionDict, Dict, Dict, Dict]:
        if not isinstance(config, dict):
            raise TypeError("Pass a python dictionary as session `config`.")
        expected_keys = ["session", "data", "network", "metrics"]
        if not all(key in config for key in expected_keys):
            raise ValueError("`config` should specify all of these fields {}".format(expected_keys))

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
    def configure_states_dir_and_randomness_sources(run_dir: str, create_run_dir_afresh: bool) -> None:
        states_dir = os.path.join(run_dir, "states")

        if create_run_dir_afresh:
            os.makedirs(states_dir, exist_ok=False)
            torch.save({"torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                        "numpy_rng_state": np.random.get_state(),
                        "random_rng_state": random.getstate()}, os.path.join(states_dir, 'rng_states.pth'))
        else:
            if not os.path.exists(states_dir):
                raise FileNotFoundError(
                    "Requested to restart from a previous run, but its states_dir to reload from is missing.")

            # TODO: The warning regarding weight_only option. Instead of pickling, just serialize them with json.
            rng_states = torch.load(os.path.join(states_dir, "rng_states.pth"))
            torch.set_rng_state(rng_states["torch_rng_state"])
            if rng_states["cuda_rng_state"] is not None:
                torch.cuda.set_rng_state(rng_states["cuda_rng_state"])
            np.random.set_state(rng_states["numpy_rng_state"])
            random.setstate(rng_states["random_rng_state"])

    @abstractmethod
    def init_datasets(self) -> Tuple[Dataset, ValidationDatasetsDict]:
        pass

    def _init_datasets(self) -> Tuple[Dataset, ValidationDatasetsDict]:
        # TODO: Save random seeds for reproducibility (e.g., if train-valid splitting is done here).

        dataset_train, datasets_valid_dict = self.init_datasets()

        if not isinstance(dataset_train, Dataset):
            raise ValueError("`dataset_train` should be an instance of `datasets.Dataset`.")
        if not isinstance(datasets_valid_dict, ValidationDatasetsDict):
            raise ValueError("`datasets_valid_dict` must be an instance of `ValidationDatasetsDict`.")
        if not datasets_valid_dict.is_valid():
            raise ValueError("Failed to create a valid `datasets_valid_dict`, an instance of `ValidationDatasetsDict`.")

        # TODO: Convert from Huggingface to `torch.utils.data.Dataset` instances.

        return dataset_train, datasets_valid_dict

    def init_dataloaders(self) -> Tuple[DataLoader, Dict[str, DataLoader]]:
        # TODO: Decide on torch vs. huggingface dataloader.
        # TODO: The dataloader for the streaming case.
        # TODO: Dataloading with partial shuffling when there are shards of data.
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
