from torchbase.utils.session import TrainingConfigSessionDict, _copy_config
from torchbase.utils.session import generate_log_dir_tag, is_custom_scalar_logging_layout_valid
from torchbase.utils.session import RandomnessGeneratorStates
from torchbase.utils.data import ValidationDatasetsDict
from torchbase.utils.metrics import BaseMetricsClass
from torchbase.utils.logger import ProgressManager, ValuesLogger, LoggableParams
from torchbase.utils.checkpoint import atomic_torch_save

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data import Dataset as TorchDataset

from datasets import Dataset

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Callable
import inspect
import os
import random
import json
import shutil

SAVED_NETWORK_NAME = "network.pth"
SAVED_RNG_NAME = "rng_states.json"
SAVED_CHECKPOINT_NAME = "checkpoint.pth"
DO_LOG_HPARAMS = True


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
        for loader in [self.dataloader_train, *self.dataloader_valid_dict.values()]:
            self._dataloader_settings(loader)

        self.network = self._init_network()

        self.optimizer = self.init_optimizer()

        self._loaded_checkpoint = None
        self.best_model_epoch = None
        self._phase = "training"
        self._validation_index = 0
        self._pending_loader_rng = {}
        self.load_network_and_optimizer_states_if_relevant(source_run_dir_tag, create_run_dir_afresh)

        self.writer = SummaryWriter(log_dir=self.run_dir)

        metrics_classes = self.init_metrics()
        self.metrics_functionals_dict = self.get_metrics_functionals_dict_from_metrics_classes(metrics_classes)

        self.progress_train = ProgressManager()
        self.loggable_train = LoggableParams({**{"loss": self.get_loss_value}, **self.metrics_functionals_dict})
        self.value_logger_train = ValuesLogger(self.loggable_train.get_names(), progress_manager=self.progress_train)

        self.progress_valid_dict = {valid_dataset_name: ProgressManager() for valid_dataset_name in
                                    self.datasets_valid_dict.names}
        self.loggable_valid_dict = {
            valid_dataset_name: LoggableParams({**{"loss": self.get_loss_value}, **self.metrics_functionals_dict})
            for valid_dataset_name in self.datasets_valid_dict.names}
        self.value_logger_valid_dict = {
            valid_dataset_name: ValuesLogger(self.loggable_train.get_names(),
                                             progress_manager=self.progress_valid_dict[valid_dataset_name]) for
            valid_dataset_name in self.datasets_valid_dict.names}

        self.best_validation_loss_dict = {
            name: (float("inf"), -2) for name, demo in
            zip(self.datasets_valid_dict.names, self.datasets_valid_dict.only_for_demo) if not demo}

        self.hparams_dict = self.init_hparams_dict()

        if not create_run_dir_afresh:
            try:
                # Initialization consumes randomness while rebuilding data and
                # the network. Restore continuation randomness only afterwards.
                self._restore_training_state()
            except Exception:
                self.writer.close()
                raise
        self._loaded_checkpoint = None

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
        missing = [key for key in expected_keys if key not in config]
        if missing:
            raise ValueError("Missing required config sections: {}".format(", ".join(missing)))
        unknown = config.keys() - set(expected_keys)
        if unknown:
            raise ValueError("Unknown config sections: {}".format(", ".join(sorted(map(str, unknown)))))
        for key in expected_keys:
            if not isinstance(config[key], dict):
                raise TypeError("The `{}` config section must be a dictionary.".format(key))
        architecture = config["network"].get("architecture")
        if not isinstance(architecture, str) or not architecture:
            raise ValueError("The network config requires a nonempty `architecture` class name.")

        return (TrainingConfigSessionDict(config["session"]), _copy_config(config["data"]),
                _copy_config(config["metrics"]), _copy_config(config["network"]))

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
            if tag_postfix is not None:
                raise ValueError(
                    "It is asked to recover an existing run (rather than creating a new one), yet a tag to post-fix a"
                    " (presumably new) run name is assigned.")

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

        if not os.path.exists(os.path.join(source_states_dir, SAVED_CHECKPOINT_NAME)):
            raise FileNotFoundError("No checkpoint.pth found. Resuming v0.1.x separate-file checkpoints is not "
                                    "supported in v0.2.x; continue with the old version or start a new experiment.")

    def add_writer_custom_scalar_logging_layout(self, layout_dict: Dict):
        if is_custom_scalar_logging_layout_valid(layout_dict,
                                                 self.datasets_valid_dict.names,
                                                 tuple(self.metrics_functionals_dict.keys())):
            self.writer.add_custom_scalars(layout_dict)
        else:
            raise TypeError("The provided `layout_dict` is not valid. Debug to see where exactly it fails.")

    @staticmethod
    def configure_states_dir_and_randomness_sources(run_dir: str, create_run_dir_afresh: bool) -> None:
        states_dir = os.path.join(run_dir, "states")

        if create_run_dir_afresh:
            rng = RandomnessGeneratorStates()
            os.makedirs(states_dir, exist_ok=False)
            rng.save(os.path.join(states_dir, SAVED_RNG_NAME))

        else:
            TrainingBaseSession.check_source_states_dir_is_valid(states_dir)
            rng = RandomnessGeneratorStates.load(os.path.join(states_dir, SAVED_RNG_NAME))
            rng.apply()

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

        return dataset_train, datasets_valid_dict

    def init_dataloaders(self) -> Tuple[StatefulDataLoader, Dict[str, StatefulDataLoader]]:
        # TODO (#10): When streaming mode gets support, dataloading and shuffling will get nuanced.
        def make_loader(dataset):
            generator = torch.Generator().manual_seed(torch.empty((), dtype=torch.int64).random_().item())
            return StatefulDataLoader(dataset, batch_size=self.config_session.mini_batch_size, shuffle=True,
                                      num_workers=self.config_session.dataloader_num_workers,
                                      collate_fn=self.dataloader_collate_function, generator=generator)

        return make_loader(self.dataset_train), {
            name: make_loader(dataset) for name, dataset in
            zip(self.datasets_valid_dict.names, self.datasets_valid_dict.datasets)}

    @staticmethod
    def dataloader_collate_function(batch: List[Any]) -> Dict[str, List[Any] | torch.Tensor]:
        return torch.utils.data.default_collate(batch)

    @abstractmethod
    def init_network(self) -> torch.nn.Module:
        pass

    def _init_network(self) -> torch.nn.Module:

        network = self.init_network()
        network = network.to(self.device)

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

    def load_network_and_optimizer_states_if_relevant(self, source_run_dir_tag: None | str,
                                                      create_run_dir_afresh: bool) -> None:
        if source_run_dir_tag is None:
            if not create_run_dir_afresh:
                raise ValueError("This cannot happen anyway.")
            return

        source_states_dir_path = os.path.join(os.path.dirname(self.run_dir), source_run_dir_tag, "states")
        checkpoint_path = os.path.join(source_states_dir_path, SAVED_CHECKPOINT_NAME)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        required = {"format_version", "network", "optimizer", "progress_train", "values_logger_train",
                    "validation", "best_validation_loss_dict", "best_model", "best_model_epoch", "rng_state",
                    "phase", "validation_index", "dataloader_train", "validation_names"}
        if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
            raise ValueError("Incomplete or unsupported checkpoint: required training state is missing.")
        if checkpoint["format_version"] != 2:
            raise ValueError("Unsupported checkpoint format version: {}".format(checkpoint["format_version"]))
        self.network.load_state_dict(checkpoint["network"])
        if not create_run_dir_afresh:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self._loaded_checkpoint = checkpoint

    def _restore_training_state(self) -> None:
        """Restore one completed iteration, including the active loop and loader positions."""
        checkpoint = self._loaded_checkpoint
        if (checkpoint["validation_names"] != list(self.datasets_valid_dict.names)
                or set(checkpoint["validation"]) != set(self.datasets_valid_dict.names)):
            raise ValueError("Checkpoint validation datasets do not match the current session.")
        if set(checkpoint["best_validation_loss_dict"]) != set(self.best_validation_loss_dict):
            raise ValueError("Checkpoint model-selection datasets do not match the current session.")
        self.best_validation_loss_dict = checkpoint["best_validation_loss_dict"]
        self.progress_train.load_state_dict(checkpoint["progress_train"])
        self.value_logger_train.load_state_dict(checkpoint["values_logger_train"])
        self._restore_dataloader(self.dataloader_train, checkpoint["dataloader_train"], self.progress_train)
        for name, state in checkpoint["validation"].items():
            self.progress_valid_dict[name].load_state_dict(state["progress"])
            self.value_logger_valid_dict[name].load_state_dict(state["values_logger"])
            self._restore_dataloader(self.dataloader_valid_dict[name], state["dataloader"],
                                     self.progress_valid_dict[name])
        self._phase = checkpoint["phase"]
        self._validation_index = checkpoint["validation_index"]
        self._check_loop_state()
        self.best_model_epoch = checkpoint["best_model_epoch"]
        if checkpoint["best_model"] is not None:
            atomic_torch_save(checkpoint["best_model"], os.path.join(self.run_dir, SAVED_NETWORK_NAME))
        RandomnessGeneratorStates.from_dict(checkpoint["rng_state"]).apply()

    @staticmethod
    def _dataloader_settings(loader) -> Dict:
        if not isinstance(loader, StatefulDataLoader) or loader.generator is None or not loader.in_order:
            raise ValueError("Recovery requires StatefulDataLoader with an explicit generator and in_order=True.")
        if loader.persistent_workers:
            raise ValueError("Recovery currently requires persistent_workers=False.")
        return {"dataset_size": len(loader.dataset), "batch_size": loader.batch_size,
                "num_workers": loader.num_workers, "drop_last": loader.drop_last,
                "persistent_workers": loader.persistent_workers, "prefetch_factor": loader.prefetch_factor}

    def _dataloader_state(self, loader, progress) -> Dict:
        return {"settings": self._dataloader_settings(loader),
                # Do not create iterators (or start workers) for inactive loaders.
                "iterator": loader.state_dict() if progress.iter_current_epoch else None,
                "generator": loader.generator.get_state()}

    def _restore_dataloader(self, loader, state, progress) -> None:
        if state["settings"] != self._dataloader_settings(loader):
            raise ValueError("Checkpoint dataloader settings do not match the current session.")
        if (state["iterator"] is None) != (progress.iter_current_epoch == 0):
            raise ValueError("Checkpoint dataloader position is inconsistent with its progress.")
        loader.generator.set_state(state["generator"])
        if state["iterator"] is not None:
            loader.load_state_dict(state["iterator"])
            self._pending_loader_rng[id(loader)] = state["generator"]

    def _dataloader_iterator(self, loader):
        generator_state = self._pending_loader_rng.pop(id(loader), None)
        if generator_state is None:
            return iter(loader)
        # StatefulDataLoader restores lazily in iter(). Reconstructing its
        # sampler and worker seed consumes RNG; that setup is not a training step.
        rng = RandomnessGeneratorStates()
        iterator = iter(loader)
        loader.generator.set_state(generator_state)
        rng.apply()
        return iterator

    def _check_loop_state(self) -> None:
        count = len(self.datasets_valid_dict.names)
        index = self._validation_index
        if (self._phase not in ("training", "validation", "selection") or type(index) is not int
                or not 0 <= index <= count
                or (self._phase == "training" and index != 0)
                or (self._phase == "selection" and index != count)):
            raise ValueError("Inconsistent checkpoint execution phase.")
        epoch = self.progress_train.epoch
        if any(type(value) is not int or value < 0 for progress in
               [self.progress_train, *self.progress_valid_dict.values()] for value in progress.state_dict().values()):
            raise ValueError("Invalid checkpoint progress counters.")
        if self._phase != "training" and (epoch == 0 or self.progress_train.iter_current_epoch != 0
                                          or self.progress_train.samples_current_epoch != 0):
            raise ValueError("Inconsistent checkpoint training progress.")
        for ind, name in enumerate(self.datasets_valid_dict.names):
            progress = self.progress_valid_dict[name]
            expected_epoch = epoch - int(self._phase == "validation" and ind >= index)
            active = self._phase == "validation" and ind == index
            if progress.epoch != expected_epoch or (not active and
                    (progress.iter_current_epoch != 0 or progress.samples_current_epoch != 0)):
                raise ValueError("Inconsistent checkpoint validation progress.")

    def _save_checkpoint(self, *, is_best: bool = False) -> None:
        """Commit one recovery point, including the selected inference model."""
        self._check_loop_state()
        best_path = os.path.join(self.run_dir, SAVED_NETWORK_NAME)
        checkpoint_path = os.path.join(self.run_dir, "states", SAVED_CHECKPOINT_NAME)
        network_state = self.network.state_dict()
        if is_best:
            best_model = network_state
            best_model_epoch = self.progress_train.epoch - 1
        elif os.path.exists(checkpoint_path):
            # The export is only a convenience copy: it may have been removed
            # or its write interrupted. Carry forward the authoritative best.
            previous = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            best_model = previous["best_model"]
            best_model_epoch = previous["best_model_epoch"]
            del previous
        else:
            best_model = None
            best_model_epoch = None
        checkpoint = {
            "format_version": 2,
            "phase": self._phase,
            "validation_index": self._validation_index,
            "validation_names": list(self.datasets_valid_dict.names),
            "network": network_state,
            "optimizer": self.optimizer.state_dict(),
            "progress_train": self.progress_train.state_dict(),
            "values_logger_train": self.value_logger_train.state_dict(),
            "dataloader_train": self._dataloader_state(self.dataloader_train, self.progress_train),
            "validation": {name: {"progress": self.progress_valid_dict[name].state_dict(),
                                  "values_logger": self.value_logger_valid_dict[name].state_dict(),
                                  "dataloader": self._dataloader_state(self.dataloader_valid_dict[name],
                                                                        self.progress_valid_dict[name])}
                           for name in self.datasets_valid_dict.names},
            "best_validation_loss_dict": self.best_validation_loss_dict,
            "best_model": best_model,
            "best_model_epoch": best_model_epoch,
            "rng_state": RandomnessGeneratorStates().to_dict(),
        }
        atomic_torch_save(checkpoint, checkpoint_path)
        self.best_model_epoch = best_model_epoch
        if is_best:
            atomic_torch_save(best_model, best_path)

    @abstractmethod
    def init_metrics(self) -> List[BaseMetricsClass] | None:
        pass

    def get_metrics_functionals_dict_from_metrics_classes(self,
                                                          metrics_classes:
                                                          List[BaseMetricsClass] | None) -> (
            Dict[str, Callable[..., Any]] | Dict[None, None]):
        if metrics_classes is None:
            return {}

        if not isinstance(metrics_classes, list) or not all(
                [isinstance(metrics_class, BaseMetricsClass) for metrics_class in metrics_classes]):
            raise TypeError(
                "The abstract `init_metrics_module` should implement a list of valid `BaseMetricsClass` instances.")

        if set(self.config_metrics.keys()) != set([_class.__class__.__name__ for _class in metrics_classes]):
            raise ValueError("The requested set of metrics classes from the metrics config does not match the "
                             "implemented metrics classes within the abstract `init_metrics_module` method.")

        metrics_functionals_dict: Dict[str, Callable[..., Any]] = {}
        for metrics_class in metrics_classes:
            requested_metrics_list = self.config_metrics[metrics_class.__class__.__name__]
            implemented_metrics = metrics_class.get_all_metric_functionals_dict()
            for metric in requested_metrics_list:
                if metric not in implemented_metrics:
                    raise ValueError("The metrics config requested to use `{}`, which is not available in the"
                                     "set of metrics implemented at `{}`".format(metric, metrics_class.__name__))

            if len(requested_metrics_list) > 0:
                _metrics_functionals_dict = metrics_class.get_metrics(requested_metrics_list)
                if set(metrics_functionals_dict.keys()) & set(_metrics_functionals_dict):
                    raise ValueError(
                        "Duplicate keys found across the metrics from different `BaseMetricsClass` instances.")
                metrics_functionals_dict = {**metrics_functionals_dict, **_metrics_functionals_dict}

        return metrics_functionals_dict

    def init_hparams_dict(self) -> Dict[str, float | int]:
        hparams_dict = {
            "learning_rate": self.config_session.learning_rate,
            "mini_batch_size": self.config_session.mini_batch_size,
            "num_epochs": self.config_session.num_epochs,
            "weight_decay": self.config_session.weight_decay,
            "num_network_params": sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        }

        return hparams_dict

    def append_hparams_dict(self, optional_hparams_dict: Dict) -> None:
        if not isinstance(optional_hparams_dict, dict) or not all(
                [isinstance(key, str) for key in optional_hparams_dict.keys()]):
            raise TypeError("Invalid `optional_hparams_dict` to be appended to the default one.")
        self.hparams_dict = {**self.hparams_dict, **optional_hparams_dict}

    def cleanup_previous_hparam_events_if_any(self):
        hparams_event_dir = os.path.join(self.run_dir, "hparams")
        if os.path.exists(hparams_event_dir):
            shutil.rmtree(hparams_event_dir)

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

    def do_one_training_epoch(self) -> None:
        # TODO (#13): Progress bar à la tqdm?
        for (ind, mini_batch) in enumerate(self._dataloader_iterator(self.dataloader_train)):
            self.do_one_training_iteration(mini_batch)
            if self.progress_train.iter_current_epoch % self.config_session.checkpoint_interval == 0:
                self._save_checkpoint()
            for param in self.value_logger_train.names:
                self.writer.add_scalar("training/{}/epochs".format(param),
                                       self.value_logger_train.average_of_epoch[param],
                                       self.progress_train.epoch + 1)

            self.print("training | epoch = {:0{}d}/{}\t"
                       "iter = {:0{}d}/{}\t"
                       "loss = {:.5f}".format(self.progress_train.epoch + 1,
                                              int(np.ceil(np.log10(self.config_session.num_epochs))),
                                              self.config_session.num_epochs,
                                              self.progress_train.iter_current_epoch,
                                              int(np.ceil(np.log10(len(self.dataloader_train)))),
                                              len(self.dataloader_train),
                                              self.value_logger_train.current_values["loss"]))

        self.progress_train.increment_epoch()

    def do_one_validation_epoch(self, valid_dataset_name: str) -> None:
        for (ind, mini_batch) in enumerate(self._dataloader_iterator(self.dataloader_valid_dict[valid_dataset_name])):
            self.do_one_validation_iteration(mini_batch, valid_dataset_name)
            if (self.progress_valid_dict[valid_dataset_name].iter_current_epoch
                    % self.config_session.checkpoint_interval == 0):
                self._save_checkpoint()

            for param in self.value_logger_valid_dict[valid_dataset_name].names:
                self.writer.add_scalar("validation-{}/{}/epochs".format(valid_dataset_name, param),
                                       self.value_logger_valid_dict[valid_dataset_name].average_of_epoch[param],
                                       self.progress_valid_dict[valid_dataset_name].epoch + 1)

            self.print("validation-{} | epoch = {:0{}d}/{}\t"
                       "iter = {:0{}d}/{}\t"
                       "loss = {:.5f}".format(valid_dataset_name,
                                              self.progress_valid_dict[valid_dataset_name].epoch + 1,
                                              int(np.ceil(np.log10(self.config_session.num_epochs))),
                                              self.config_session.num_epochs,
                                              self.progress_valid_dict[valid_dataset_name].iter_current_epoch,
                                              int(np.ceil(
                                                  np.log10(len(self.dataloader_valid_dict[valid_dataset_name])))),
                                              len(self.dataloader_valid_dict[valid_dataset_name]),
                                              self.value_logger_valid_dict[valid_dataset_name].current_values["loss"]))

        self.progress_valid_dict[valid_dataset_name].increment_epoch()

    def train(self):
        # Preserve the initialized state too, so an interrupted first epoch can restart.
        if not os.path.exists(os.path.join(self.run_dir, "states", SAVED_CHECKPOINT_NAME)):
            self._save_checkpoint()

        while self.progress_train.epoch < self.config_session.num_epochs or self._phase != "training":
            if self._phase == "training":
                if self.progress_train.iter_current_epoch == 0:
                    self.value_logger_train.reset_epoch()
                self.do_one_training_epoch()
                self.print("\t === averaged over this epoch = {:.5f} === ".format(
                    self.value_logger_train.average_of_epoch["loss"]))
                self._phase = "validation"
                self._save_checkpoint()

            while self._validation_index < len(self.datasets_valid_dict.names):
                self.print("\n")
                valid_dataset_name = self.datasets_valid_dict.names[self._validation_index]

                if self.progress_valid_dict[valid_dataset_name].iter_current_epoch == 0:
                    self.value_logger_valid_dict[valid_dataset_name].reset_epoch()
                self.do_one_validation_epoch(valid_dataset_name)
                self.print("\t === averaged over this epoch = {:.5f} === ".format(
                    self.value_logger_valid_dict[valid_dataset_name].average_of_epoch["loss"]))
                self._validation_index += 1
                if self._validation_index == len(self.datasets_valid_dict.names):
                    self._phase = "selection"
                self._save_checkpoint()

            # Individual best records must not change until every validation set
            # has finished, so interrupted validation cannot change the votes.
            self._phase = "selection"
            i_epoch = self.progress_train.epoch - 1
            vote_for_epoch_as_successful: List[bool] = []
            for ind_set, valid_dataset_name in enumerate(self.datasets_valid_dict.names):
                if not self.datasets_valid_dict.only_for_demo[ind_set]:
                    best_validation_loss, best_previous_epoch = self.best_validation_loss_dict[valid_dataset_name]
                    current_epoch_valid_loss = self.value_logger_valid_dict[valid_dataset_name].average_of_epoch["loss"]
                    self.print(
                        ".. Best previous loss for Validation-{}  was {:.5f} "
                        "(at epoch {}/{}) and is now {:.5f}.".format(valid_dataset_name, best_validation_loss,
                                                                     best_previous_epoch + 1,
                                                                     self.config_session.num_epochs,
                                                                     current_epoch_valid_loss), end=" --> ")
                    if current_epoch_valid_loss < best_validation_loss:
                        self.best_validation_loss_dict[valid_dataset_name] = (current_epoch_valid_loss, i_epoch)
                        self.print("Voting FOR this model.")
                        vote_for_epoch_as_successful.append(True)
                    else:
                        self.print("Voting AGAINST this model.")
                        vote_for_epoch_as_successful.append(False)

            is_best = all(vote_for_epoch_as_successful)
            self._phase = "training"
            self._validation_index = 0
            self._save_checkpoint(is_best=is_best)

            if is_best:
                if DO_LOG_HPARAMS:
                    self.cleanup_previous_hparam_events_if_any()
                    self.writer.add_hparams(hparam_dict=self.hparams_dict,
                                            metric_dict={"best_loss_{}".format(_k): _v for _k, (_v, _i) in
                                                         self.best_validation_loss_dict.items()}, run_name="hparams")

                self.print("SAVED the model at the epoch {}/{} ..".format(
                    i_epoch + 1, self.config_session.num_epochs))
            self.print("\n")

        self.print("\nThis is the end of training and validation.")
        if self.best_model_epoch is not None:
            self.print("The best model was saved at epoch {}/{}.".format(self.best_model_epoch + 1,
                                                                             self.config_session.num_epochs))
            # TODO: JIT-compilation of the final saved model.2

        self.writer.close()

    def __call__(self):
        try:
            self.train()
        finally:
            self.writer.close()
