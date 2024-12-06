import torch
import os.path
from typing import Tuple


def load_network_from_state_dict_to_device(network: torch.nn.Module, state_dict_path: str | None = None,
                                           device: torch.device = torch.device("cpu")) -> torch.nn.Module:
    if not isinstance(network, torch.nn.Module):
        raise TypeError("The `network` should be an instance of `torch.nn.Module`.")
    if state_dict_path is not None and not os.path.exists(state_dict_path):
        raise FileNotFoundError("The `state_dict_path` is specified but does not exist.")

    if state_dict_path is not None:
        network.load_state_dict(torch.load(state_dict_path, map_location=device, weights_only=True))
    return network.to(device)


def jit_script_compile(network: torch.nn.Module,
                       example_inputs: None | Tuple[torch.Tensor, ...] = None) -> torch.nn.Module:
    if example_inputs is not None:
        if not isinstance(example_inputs, tuple) or any(
                not isinstance(_input, torch.Tensor) for _input in example_inputs):
            raise TypeError("`example_inputs`, if specified, should be a tuple of `torch.Tensor` objects.")
        example_inputs = [example_inputs]

    network = network.eval()

    torch.jit.enable_onednn_fusion(True)
    with torch.no_grad():
        with torch.jit.optimized_execution(should_optimize=False):
            scripted_module = torch.jit.script(network, example_inputs=example_inputs)
    torch.jit.freeze(scripted_module)   # TODO: A bit arbitrary these choices, but also because jit is not mature.
    torch.jit.enable_onednn_fusion(False)

    return scripted_module
