"""The README's binary-image experiment, with a runnable start/resume interface.

Run from the repository root with:
    uv run --locked python -m examples.image_reconstruction
See examples/README.md for the experimental setup and recovery assumptions.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from torchvision.transforms import RandomRotation

from torchbase import TrainingBaseSession
from torchbase.utils import BaseMetricsClass, ValidationDatasetsDict, split_iterables
from torchbase.utils.metrics_instances import BinaryClassificationMetrics, ImageReconstructionMetrics


def get_config() -> dict:
    """Settings for one experiment; edit these without changing the session logic."""
    return {
        "session": {
            "device_name": "cpu",
            "num_epochs": 10,
            "mini_batch_size": 6,
            "learning_rate": 0.01,
            "checkpoint_interval": 2,
        },
        "data": {
            "num_images": 20,
            "image_size": (32, 32),
            "split_portions": (0.8, 0.2),
            "rotation_degrees": 10,
        },
        "metrics": {
            "BinaryClassificationMetrics": ["precision_micro", "f1_score_micro"],
            "ImageReconstructionMetrics": ["psnr"],
        },
        "network": {"architecture": "MyFavoriteNetwork", "num_ch": 2, "num_layers": 3},
    }


class MyFavoriteNetwork(torch.nn.Module):
    """Preserve image shape and return logits, as required by BCEWithLogitsLoss."""

    def __init__(self, num_ch: int, num_layers: int):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(torch.nn.Conv2d(num_ch, num_ch, 3, padding=1))
            if i < num_layers - 1:
                layers.append(torch.nn.ReLU())
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MyTrainingSession(TrainingBaseSession):
    def init_datasets(self) -> tuple[Dataset, ValidationDatasetsDict]:
        """Reconstruct binary images; compare ordinary and rotated validation data.

        Rotation is applied once during setup, not randomly at each iteration.
        The session restores its initialization RNG before rebuilding these data
        on recovery, including the random split and rotations.
        """
        images = [torch.rand(self.config_network["num_ch"], *self.config_data["image_size"]).round()
                  for _ in range(self.config_data["num_images"])]
        data_train, data_valid = split_iterables(images, portions=tuple(self.config_data["split_portions"]))
        degrees = self.config_data["rotation_degrees"]
        rotate = RandomRotation(degrees=(-degrees, degrees))

        def augment(example):
            return {"image": rotate(example["image"])}

        dataset_train = Dataset.from_dict({"image": data_train}).with_format("torch").map(augment)
        dataset_valid = Dataset.from_dict({"image": data_valid}).with_format("torch")
        dataset_valid_aug = Dataset.from_dict({"image": data_valid}).with_format("torch").map(augment)

        # Log the training data under evaluation conditions, but do not let it vote.
        return dataset_train, ValidationDatasetsDict(
            datasets=(dataset_train, dataset_valid, dataset_valid_aug),
            only_for_demo=(True, False, False), names=("train", "valid", "valid-aug"))

    def init_network(self) -> torch.nn.Module:
        return MyFavoriteNetwork(self.config_network["num_ch"], self.config_network["num_layers"])

    def forward_pass(self, mini_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Loss and metrics select their keyword arguments from this dictionary."""
        target_image = mini_batch["image"].to(self.device)
        output_image = self.network(target_image)
        return {"output": output_image, "target": target_image,
                "gt_for_metrics": target_image, "predictions_for_metrics": output_image.sigmoid()}

    def loss_function(self, *, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Targets are binary; the loss takes logits, while metrics take probabilities.
        return torch.nn.functional.binary_cross_entropy_with_logits(output, target)

    def init_metrics(self) -> list[BaseMetricsClass]:
        """Register groups here; config['metrics'] selects the scores to log."""
        return [
            BinaryClassificationMetrics(keyword_maps={
                "gt_for_metrics": "binary_ground_truth", "predictions_for_metrics": "prediction_probabilities"}),
            ImageReconstructionMetrics(keyword_maps={
                "gt_for_metrics": "target_image", "predictions_for_metrics": "output_image"}),
        ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    run = parser.add_mutually_exclusive_group()
    run.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Parent directory for a new run.")
    run.add_argument("--resume", type=Path, help="Existing run directory to continue, not its checkpoint file.")
    parser.add_argument("--epochs", type=int, help="Total number of epochs, not additional epochs.")
    args = parser.parse_args(argv)
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive.")

    if args.resume is None:
        config = get_config()
        session_kwargs = {"runs_parent_dir": str(args.runs_dir)}
    else:
        run_dir = args.resume.resolve()
        # Each recovery saves another config snapshot, including an increased epoch limit.
        configs = sorted(run_dir.glob("config*.json"))
        if not configs:
            parser.error("No saved configuration found in {}.".format(run_dir))
        with configs[-1].open() as file:
            config = json.load(file)
        session_kwargs = {"runs_parent_dir": str(run_dir.parent), "source_run_dir_tag": run_dir.name,
                          "create_run_dir_afresh": False}
    if args.epochs is not None:
        config["session"]["num_epochs"] = args.epochs

    session = MyTrainingSession(config, **session_kwargs)
    print("Run directory: {}".format(session.run_dir), flush=True)
    session()
    return session


if __name__ == "__main__":
    main()
