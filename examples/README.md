# Image reconstruction

[image_reconstruction.py](image_reconstruction.py) provides a runnable version of the
example in the main README with more details. The idea is to show you how `torchbase` could help you in shaping your
experiments. So it's still a toy example with minimal dependency, rather than a working image recon app.

## Run it

```sh
uv sync --locked
uv run --locked python -m examples.image_reconstruction
```

The development dependencies include `torchvision`, which this example uses for
rotation. It is not a runtime dependency of `torchbase`. The example stays outside
the library package; it is available in the repository, not as an installed CLI.

The default experiment runs ten epochs on CPU, with zero dataloader workers. To
run fewer epochs or choose where to keep the experiment:

```sh
uv run --locked python -m examples.image_reconstruction --epochs 2 --runs-dir runs
uv run --locked tensorboard --logdir=runs
```

The program prints the run directory before training begins. Each new invocation
without `--resume` creates a new time-tagged run. Run directories contain experiment
data and should not be committed; the default `runs/` directory is git-ignored.

## The experimental setup

The five session methods have the same responsibilities as in the README:

- `init_datasets` generates 20 binary images with shape `(2, 32, 32)`, splits them
  into 16 training and 4 validation images, and formats the datasets as tensors.
- `init_network` creates three shape-preserving convolutional layers. The final
  layer has no activation, so the network returns unrestricted logits.
- `forward_pass` uses the image as both input and target. It returns logits for
  the loss and sigmoid probabilities for the metrics.
- `loss_function` applies mean-reduced binary cross-entropy to each pixel.
- `init_metrics` registers binary classification and reconstruction metrics.
  Configuration selects precision, F1 and PSNR from these groups.

`get_config()` keeps the experimental settings separate from this setup. It adds
the image count and rotation angle as configurable values, and sets
`checkpoint_interval=2` so even this short experiment saves inside an epoch.
The library default is 100. Edit these settings for a **new** experiment; recovery
loads saved settings instead of using the current contents of `get_config()`.

### Augmentation and validation

Rotation is applied once when building the datasets with `Dataset.map`. Nearest-
neighbor rotation preserves the binary labels. The resulting datasets do not
change at each iteration; this example does not demonstrate random worker-time
augmentation or denoising with different input and target images.

There are three evaluation datasets:

| Name        | Data                                                               | Participates in model selection? |
|-------------|--------------------------------------------------------------------|----------------------------------|
| `train`     | The augmented training images, evaluated after training each epoch | No (`only_for_demo=True`)        |
| `valid`     | Unaugmented validation images                                      | Yes                              |
| `valid-aug` | A rotated version of those same validation images                  | Yes                              |

Each participating validation dataset tracks its own lowest loss. A model is
selected only when **both** validation losses improve their respective records
in the same epoch. They are not combined into one average loss. The demo dataset
is logged but does not vote, and neither validation variant is a held-out test set.

### Reading the metrics

TensorBoard separates mini-batch values (`iterations`), sample-weighted means of
batch scores (`batch_means`), and supported whole-epoch scores (`epochs`).

Binary precision and F1 have whole-epoch scores computed from confusion counts
over all pixels. Here the micro scores both equal pixel accuracy; these are not
positive-class-only precision and F1. PSNR remains a batch statistic, so its
`batch_means` curve is not a PSNR recomputed over the whole dataset.
`loss/epochs` is the sample-weighted mean of the mean-reduced batch losses.

Training metrics include predictions from successive updates of the network.
The `validation-train` curves instead evaluate the training images using the
fixed network at the end of the epoch. These are different measurements.

## Recover an experiment

Use the **run directory** printed by the program, not `states/checkpoint.pth`:

```sh
uv run --locked python -m examples.image_reconstruction --resume runs/YOUR_RUN_TAG
```

Once the initial checkpoint exists, you can interrupt with Ctrl+C and run this
command afterwards. The experiment may finish before you interrupt it; resuming a completed run performs
no more iterations. To extend it, specify a larger total number of epochs:

```sh
uv run --locked python -m examples.image_reconstruction --resume runs/YOUR_RUN_TAG --epochs 15
```

This means 15 epochs in total, not 15 more. The command reads the latest saved
`config*.json` snapshot, including an epoch limit supplied during a previous
recovery. Do not edit the saved configuration files or change the code, dataset
contents, batch size, validation setup or dependencies while recovering a run.

With the example's interval of two, a successful checkpoint after training batch
2 lets recovery continue at batch 3. If batch 3 finished but its following save
did not, that batch is repeated. Checkpoints are also written before the first
epoch and at phase boundaries. Progress is saved separately for training and
each validation dataset, including the demo dataset.

Recovery restores the optimizer, latest network, dataloader positions, random
states, progress, accumulated metrics and model-selection records. The initial
random state first reconstructs the same generated images, split and rotations;
the continuation state then restores training randomness. This is tested on CPU
with unchanged code and dependencies, not a guarantee across platforms or
arbitrary custom augmentation pipelines. 

The latest recovery state is in `states/checkpoint.pth`. The separate `network.pth`
contains the **selected best model**, which may differ from the latest network.
Temporary checkpoint writes are atomically replaced, assuming one writer and a
filesystem supporting atomic replacement. TensorBoard may retain events from
iterations repeated after recovery.

## Load the selected model

From the repository root, use the configuration from the run to reconstruct the
network before loading its weights:

```python
import json
from pathlib import Path
import torch
from examples.image_reconstruction import MyFavoriteNetwork

run_dir = Path("runs/YOUR_RUN_TAG")
config = json.loads((run_dir / "config.json").read_text())
network = MyFavoriteNetwork(config["network"]["num_ch"], config["network"]["num_layers"])
network.load_state_dict(torch.load(run_dir / "network.pth", map_location="cpu", weights_only=True))
network.eval()
with torch.no_grad():
    image = torch.rand(1, config["network"]["num_ch"], *config["data"]["image_size"]).round()
    probabilities = network(image).sigmoid()
```

## Testing your experimental setup

It is a good idea to write unit tests for your ML experiments, particularly for
the experimental setup that you intend to reuse. `torchbase` helps with this by
separating the datasets, network, forward pass, loss and metrics into methods
that you can exercise independently. You do not need to run a whole training
experiment to find a wrong tensor shape or a detached loss.

[tests/test_image_reconstruction.py](tests/test_image_reconstruction.py) shows
how to do this with Python's standard `unittest`. Each test creates a small CPU
session with controlled randomness and a temporary run directory. It closes the
writer and restores random states afterwards. No new testing library is needed.

Run it from the repository root:

```sh
uv run --locked python -m unittest examples.tests.test_image_reconstruction -v
```

The tests check dataset shapes, types and binary targets; which validation sets
participate in model selection; the forward-pass outputs; loss values and gradient
flow; metrics on handcrafted predictions; and one optimizer step with its progress
and logging updates. These are tests of the setup's expected behavior, not a
requirement to achieve a certain accuracy after training.

When adapting the example, adapt these expectations too. A new dataset may need
checks for label ranges and train/validation leakage. A different network may
need checks on output dimensions, and a custom loss or metric should have a small
case with a known answer. The binary-label and shape assertions here are specific
to this example, not rules imposed by `torchbase` on every experiment.

This is the same separation discussed in the main README: version-control and
test the reusable experimental setup, while saving each run's configuration and
results separately. Tests help establish that the implementation behaves as
intended; they do not establish scientific validity or how well it generalizes.

### End-to-end checks

The focused suite above includes one single-iteration integration check. The
repository also has longer tests for the complete run and checkpoint recovery:

```sh
uv run --locked python -m unittest tests.test_readme tests.test_examples -b
```

This command includes the focused suite above and runs the README snippets and
the runnable example. The recovery tests interrupt training and evaluation between
checkpoint saves, then compare the recovered state with an uninterrupted run. The interruption is injected by
the tests; it is not a special mode in the example. CI runs these tests against
the built wheel, keeping the examples outside the library package.
