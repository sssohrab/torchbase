# `torchbase`

A minimalist base for ML projects in PyTorch


![Coverage](https://codecov.io/gh/sssohrab/torchbase/branch/main/graph/badge.svg)

## Installation

Simply pip-install in the environment where your ML projects lies, paying attention to the requirements:

```shell
pip install torchbase
```

## How it works

`torchbase` is built around the `TrainingBaseSession` class. Import it with:

```python
from torchbase import TrainingBaseSession
```

This is an abstract class, meaning that you should subclass it and implement some of its methods yourself. In
particular, you should implement how you would like to shape your training and validation databases, what network you
want to train, how a forward pass of a mini-batch of the data to the network looks like, what the loss function is and
which arguments it requires and whether or not you would like to log certain values with certain metrics.

This may look something like:

```python
from torchbase import TrainingBaseSession
from torchbase.utils import ValidationDatasetsDict, BaseMetricsClass, split_iterables

from torchbase.utils.metrics_instances import BinaryClassificationMetrics, ImageReconstructionMetrics

import torch
from datasets import Dataset

from typing import Tuple, Dict, Any, List


class MyTrainingSession(TrainingBaseSession):
    def init_datasets(self) -> Tuple[Dataset, ValidationDatasetsDict]:
        """
        Here you define and return your training dataset, as well as validation datasets. 
        They all should be instances of the Huggingface's dataset library, i.e., `datasets.Dataset` objects.
        
        The validation datasets should be wrapped around a `torchbase.ValidationDatasetsDict` object. This forces
        you to do some extra work, but having multiple validation datasets can be very useful, e.g., to monitor 
        the effect of data augmentation. 
        
        You can mark some of the validation datasets as "only for demo". In this case they will not be used for 
        the actual validation process but only logged. This can be useful, e.g., if you are curious how your network
        behaves on external datasets, or the test dataset that you are not allowed to touch, or even the training 
        dataset itself that you would want to monitor and compare under the exact conditions as the validation set 
        (i.e., when a training epoch has finished).
        
        Note that you can have access to a config file to set some parameters outside of this code block if you need. 
        This is explained right after this code block.
        """

        def augment(image):
            from torchvision.transforms import RandomRotation
            return RandomRotation(degrees=(-10, 10))(image)

        data_train, data_valid = split_iterables([torch.randn((self.config_data["image_size"])) for _ in range(20)],
                                                 portions=self.config_data["split_portions"],
                                                 shuffle=True)

        dataset_train = Dataset.from_dict({"image": data_train}).map(lambda x: augment(x))
        dataset_valid = Dataset.from_dict({"image": data_valid})
        dataset_valid_aug = Dataset.from_dict({"image": data_valid}).map(lambda x: augment(x))

        return dataset_train, ValidationDatasetsDict(datasets=(
            dataset_train, dataset_valid, dataset_valid_aug), only_for_demo=(True, False, False),
            names=("train", "valid", "valid-aug"))

    def init_network(self) -> torch.nn.Module:
        class MyFavoriteNetwork(torch.nn.Module):
            def __init__(self, num_ch, num_layers):
                super().__init__()
                layers = []
                for _ in range(num_layers):
                    layers.append(torch.nn.Conv2d(num_ch, num_ch, 3))
                    layers.append(torch.nn.ReLU())

                self.layers = torch.nn.Sequential(*layers)

            def forward(self, x):
                return self.layers(x)

        return MyFavoriteNetwork(self.config_network["num_ch"], self.config_network["num_layers"])

    def forward_pass(self, mini_batch: Dict[str, Any | torch.Tensor]) -> Dict[str, Any | torch.Tensor]:
        """
        Here you specify a forward pass of the data to the network, however you like to prepare or pass it, and from
        the network output(s) before being consumed by the loss function and the optional loggable metrics. 
        
        The mini_batch, expected to be a dictionary, is created by the `torch.utils.data.Dataloader` object that the 
        `TrainingBaseSession` internally creates for the train and validation datasets. For this, you may need to 
        re-implement the `self.dataloader_collate_function(batch: List[Any]) -> Dict[str, List[Any] | torch.Tensor]`
        of the class, if the default behavior is not appropriate for you.
        
        Importantly, the `forward_pass` function should return a dictionary with keys that the loss function and
        metrics understand.
        """
        input_image = mini_batch["image"].to(self.device)
        target_image = mini_batch["image"].to(self.device)

        output_image = self.network(input_image)

        return {"output": output_image, "target": target_image,
                "gt_for_metrics": target_image.sigmoid().round(), "predictions_for_metrics": output_image.sigmoid()}

    def loss_function(self, *, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        The loss function only receives keyword-based arguments. This should be a subset of what the above 
        `self.forward_pass` method returns with the keys matching exactly.
        """
        criterion = torch.nn.BCEWithLogitsLoss()

        return criterion(output, target)

    def init_metrics(self) -> List[BaseMetricsClass] | None:
        """
        Other than the loss function that is always logged, you may sometimes need to log other parameters or
        some metrics based on them. This function should return a list of subclasses of the `utils.BaseMetricsCLass`, which
        is a way to encapsulate similar metrics together in a unified way. Several subclasses of this class are implemented 
        for common tasks such as binary classification or image segmentation. You can create your own metrics by 
        subclassing the `BaseMetricsClass` as a wrapper around concrete metric implementations such as `sklearn.metrics`.
        
        Similarly to the `loss_function` above, this function receives all its inputs from the key-value pairs of the 
        `forward_pass`. The `keyword_maps` argument of the `BaseMetricsClass` can translate the keywords between the two,
        in case they are different.
        
        While you specify the subclasses of `BaseMetricsClass` in this function, which are essentially groups of similar 
        metrics around one particular task, which metrics from every family to log for any particular experiment run is
        specified under the `metrics` field of the config dictionary, as is shown in the example of the next code block.
        So the metrics in the config will specify a task, and a list of concrete metrics that are already implemented 
        for each. 
        """
        metrics_class_binary_classification = BinaryClassificationMetrics(keyword_maps={
            "gt_for_metrics": "binary_ground_truth",
            "predictions_for_metrics": "prediction_probabilities"})

        metrics_class_image_reconstruction = ImageReconstructionMetrics(keyword_maps={
            "gt_for_metrics": "target_image",
            "predictions_for_metrics": "output_image"
        })

        metrics_classes_list = [metrics_class_binary_classification, metrics_class_image_reconstruction]
        return metrics_classes_list
```

Before instantiating this class and creating the session, you will need to set up a configuration dictionary that could
look like this:

```python
config = {
    "session": {
        "device_name": "cpu",
        "num_epochs": 10,
        "mini_batch_size": 6,
        "learning_rate": 0.01
    },
    "data": {
        "image_size": (32, 32),
        "split_portions": (0.8, 0.2)
    },
    "metrics": {
        "BinaryClassificationMetrics": [
            "precision_micro", "f1_score_micro"
        ],
        "ImageReconstructionMetrics": [
            "psnr"
        ]
    },
    "network": {
        "architecture": "MyFavoriteNetwork",
        "num_ch": 2,
        "num_layers": 3
    }
}
```

- The `"session"` field configures general attributes of the training session. All subfields in the above example are
  necessary, and some others exist with default values.
- The `"data"` field does not have any necessary subfield, but you may specify any configuration that
  the `self.init_datasets()` method would need to access, e.g., the dataset splitting portions, data augmentation
  hyperparameters or any other setting that your datasets may require.
- The `"metrics"` field has no necessary subfield and can be left empty. It can be used to specify configurations
  related to anything you want to log other than the loss function and any metrics related to them.
  The `self.init_metrics()` method has access to this field.
- The `"network"` field has only the `"architecture"` subfield as necessary, where you should just specify the name of
  your network. Here you can specify any hyperparameter your network instantiation would need.

Why do we need this separate config dictionary while we can have it all implemented within the abstract methods of the
class? The idea is that the part of the implementation that is less likely to change very frequently will go to the
methods implementations as python code. You would want to version-control this part using `git` as the main components
of your experimental setup. On the other hand, whatever you want to fiddle with frequently at every experiment should go
to the config dictionary. You don't want to git-commit every time you change the learning rate or add more layers to
your network. Yet you want every single experiment to be exactly reproducible. Therefore, the config dictionary
corresponding to every experiment is saved as a `json` file and could be accessed and inspected later if needed.

Having implemented the abstract methods of session class and once the config dictionary is set up, you are ready to
initialize your training session as:

```python
session = MyTrainingSession(config)
```

This will create a time-tagged directory where the artefacts of the experiment like the config file, the randomness
seeds, the network weights, the intermediate logging variables or the tensorboard-formatted loggable parameters are
saved.

To actually launch the training process itself, you will simply call your session object as:

```python
session()
```

This will run all the training and validation loops following the standard deep learning practice and accordingly to
your specified setup, and will keep saving the network with the lowest validation loss it would have seen during the
execution of the training epochs.

```shell
tensorboard --logdir="path/to/some/parent/dir/of/my/experiments"
```

Other than some reports generated at the standard output, running a tensorboard service as above referring to the parent
directory of your experiment runs will launch the web-service from where you could track your experiments, check the
logged parameters, judge whether you are going over- or under-fit, or decide the best set of hyperparameters across
experiments.

For various reasons, you may sometimes want to suspend the training process before it is finalized. `torchbase` provides
you with the possibility to take-over from a past experiment and recover (almost) exactly what was going on:

```python
the_recovered_session = MyTrainingSession(same_config_as_before,
                                          source_run_dir_tag=the_tag_to_the_suspended_experiment,
                                          create_run_dir_afresh=False)

the_recovered_session()  # This will continue the suspended session where it stopped without creating a new run dir.
```

Alternatively, passing the flag `create_run_dir_afresh=True`, but still specifying a `source_run_dir_tag` will create a
new experiment starting from scratch but only initializing the weights of the network with the previously-trained ones
from the source run.

As a general note, other than the config parameters which are accessible to all member methods, you may want to pass
some new members to the class accessible to all methods. In this case, you can re-implement the `__init__` method with
new variables and call the one from the super-class (`TrainingBaseSession`) accordingly, or even add new methods if you
need:

```python
class MyTrainingSession(TrainingBaseSession):
    def __init__(self, config, some_new_member_variable):
        super().__init__(config)
        self.new_variable = some_new_member_variable

    def some_new_method(self):
        pass
```

Currently, the `torchbase` project has no documentation other than this readme. The best way to learn about its
different features is to look at the unit-tests, or just looking at the source code.

## The idea behind

This project has the following ideas:

### I. Automating repetitive parts, but only to the right amount:

Deep learning is highly modular. The use-cases and applications vary requiring you to reformulate certain modules, yet
some others get repeated over and over again with minimal to no change. This, in fact, is one of the most prominent
reasons behind its widespread adoption: that you do not need to start from scratch when you change your problem domain
from say, computer vision to something very different like text. Instead, you would keep most of your project modules
almost the same, but only change certain details about some of them and accordingly to the new task or domain.

In practice, this means a lot of your ML projects are going to look very similar to each other in certain ways. From
the software development point of view though, this raises an important challenge on how to avoid repeating yourself
across projects by maximizing code re-use. Otherwise, things could get very boring and certainly very error-prone. You
don't want, e.g., to copy-paste that data loading piece of code you wrote at your last job yet once more for your new
project. Or you don't like to take those training-validation loops they taught you at school from that sloppy Jupyter
notebook homework block and repeat it in all your projects every single time, only with some little adjustments here and
there.

At the same time though, automating these steps to avoid code redundancy can also be a very tricky business, as the
changes within different ML projects or different trials within the same project may not merely be changing some
known-in-advance hyperparameters of them. Remember the last automation you did where you would launch that sort of
`python train.py -num_layers=3 -lr=0.001` command with all those flags and then all the steps of training, validation
and evaluation would be taken care of? This may no longer hold as nicely, as you should now change the whole code logic
ever since your new loss function now requires to access some of your network internals. In other words, you took the
structure of your project too seriously at its last version, and you automated it too much only to realise that it is
not flexible enough to serve your future use-cases.

`torchbase` is an attempt to automate only the very most repetitive elements within deep learning projects, i.e., the
common denominator to most ML projects, but stay agnostic to others by providing the user with sufficient flexibility on
however they want to implement their project's logic.

### II. Separation of the experimental setup from the experiment:

Parallel to the discussion above and thinking abstractly, you can consider a whole hierarchy of operations performed
and pieces of code running during a typical deep learning experiment. At the lower side of the hierarchy are the
operations like tensor manipulations or matrix multiplications requiring very rigorous implementations and the best
practices of software development and algorithmic efficiency. Then there is the computational graphs, normalizations,
the neural modules and architectures, etc.

Fortunately, a lot of these abstraction layers are taken care of by deep learning frameworks such as PyTorch (which are
themselves based on various lower-level resources), ensuring that implementations are efficient and properly tested and
verified. This lifts the burden of the tedious works from the users allowing them to focus on their specific use-cases.

At the other end of the hierarchy though, there is your tentative prototypes and trials to experiment with different
configurations and hyperparameters. Obviously, you do not need the same level of rigor and care
here as you did for the lower-level stuff. While you would still want to stick to good software development practice to
maintain your project's code base with proper versioning, you don't want, e.g., to git-commit the number of
convolutional blocks of that candidate architecture just for this run of the experiment, knowing that you are likely to
change it a couple of times before it gets a stable value.

Since deep learning experiments can vary largely across different applications, there cannot be a standard solution for
the more user-facing layers as there is for the lower-level ones (i.e., frameworks like PyTorch). Nevertheless, it still
is very useful to think of every project as various layers of abstraction and treat each layer accordingly to its
nature.

`torchbase` helps you shape your projects in a more modular way. It sits somewhere between PyTorch and your project code
that you write yourself, and furthermore pushes your code to be developed in a layered way. Think of it as a sort of
**experimentation framework** that comes before the **experimental setup** and the **experiment run** itself. While you
write the latter two yourself, it still pushes you to make this separation explicit.

In particular and on top of PyTorch itself, basic operations such as training loops and the validation logic which are
common to most deep learning experiments are taken care of by `torchbase` without the need for user intervention. The
user, however, is responsible for setting up and maintaining their own experimental setup. This would include the exact
specification of the forward pass of a data mini-batch to a user-defined network, the loss function and the data itself.
Since the experimental setup is meant to be re-used multiple times, some level of version controlling, testing and
maintenance would be needed from the user side. Finally, as for the experiment itself, the specification of the exact
configurations are supposed to be regarded as separate config values that would generate the outcomes of the experiment.
These outcomes, along with the configurations used and other artefacts are saved under an experiment-specific
**run directory**. These run directories, while not version-controlled, can later be used along with the
version-controlled experimental setup to repeat and reproduce an experiment.

### III. Enforcing best practices without overwhelming users with features:

As in any experimental field, successful deep learning trials require certain best practices to be respected. You would
want, e.g., to systematically log your experimental outcomes and regularly monitor the progress of the training process
based on them. You would want to be able to easily compare different hyperparameter candidates and the effect of each.
You would want to be very careful about the data types of every data field you pass to avoid unexpected behaviors before
they appear. You would definitely want to care about repeatability and reproducibility of your experiments in the
future.

`torchbase` helps you with all of these and shapes your projects accordingly to systematically follow good practice.
Unlike many existing alternatives, however, things are kept relatively minimalist. Beyond experiment tracking and
management, it does not offer, e.g., any feature for serving and deploying your trained models, or it does not sell you
any solution
to productionize your ML models, as it does not care about which cloud provider you use, if you choose to do so.

While being minimalist and hence not pushing the user to adopt features they don't always need, this project still
involves certain choices and introduces certain limitations. Notably, it has the following assumptions:

- That you use PyTorch.
- That all datasets are introduced as objects from the [datasets](https://github.com/huggingface/datasets) library.
- That you are willing to run a [tensorboard](https://github.com/tensorflow/tensorboard) service to monitor your
  experiment logs.
- That you understand what `torchbase` is doing under the hood by checking on the test cases and reviewing its source
  code.

## Licence

This project is licenced under the Apache License 2.0, as is included into the repository, and leverages the following
open source libraries:

- [PyTorch](https://pytorch.org/) - BSD 3-Clause License
- [Hugging Face `datasets`](https://github.com/huggingface/datasets) - Apache License 2.0
- [TensorBoard](https://github.com/tensorflow/tensorboard) - Apache License 2.0
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) - BSD 3-Clause License

Each library is used in accordance with its license.

As a disclaimer, this project is **not affiliated with, endorsed by, or officially part of the PyTorch ecosystem** or
any other library mentioned above.

## Contributing

This is an early stage project and welcomes community contributions and improvement proposals. Feel free to:

- Open a [GitHub issue](https://github.com/sssohrab/torchbase/issues).
- Submit a pull request with your proposed changes.
- Reach out directly to the owner of the project via [email](mailto:sohrab.ferdowsi@gmail.com).


