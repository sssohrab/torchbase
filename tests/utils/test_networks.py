from torchbase.utils.networks import load_network_from_state_dict_to_device, jit_script_compile

import torch
from torchvision import models

import unittest
import os
import tempfile


class JITScriptCompileUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if torch.cuda.is_available():
            cls.device = torch.device("cuda:0")
        else:
            cls.device = torch.device("cpu")

        cls.network = models.mobilenet_v3_small().to(cls.device)  # Or, whatever

        cls.input_tensor = torch.randn(1, 3, 32, 32).to(cls.device)
        cls.input_tensor_as_example_for_jit = (torch.rand_like(cls.input_tensor).to(cls.device),)

        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.state_dict_path = os.path.join(cls.temp_dir.name, 'network.pth')
        cls.compiled_path = os.path.join(os.path.dirname(cls.state_dict_path), "network_script.pt")

        torch.save(cls.network.state_dict(), cls.state_dict_path)

        network = load_network_from_state_dict_to_device(cls.network, cls.state_dict_path, device=cls.device)
        scripted_module = jit_script_compile(network, cls.input_tensor_as_example_for_jit)

        scripted_module.eval().save(cls.compiled_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_compiled_file_exists(self):

        self.assertTrue(os.path.exists(self.compiled_path))

        scripted_module = torch.jit.load(self.compiled_path)
        self.assertIsInstance(scripted_module, torch.jit.ScriptModule)
        self.assertIsInstance(scripted_module, torch.nn.Module)

    def _test_jit_out_close_to_net_out(self, scripted_module: torch.nn.Module):
        with torch.no_grad():
            output_ground_truth = self.network(self.input_tensor)

        with torch.no_grad():
            output_jit = scripted_module(self.input_tensor)

        self.assertTrue(torch.allclose(output_ground_truth, output_jit, rtol=1e-4, atol=1e-5),
                        "The output of the JIT-compiled net is not close enough to "
                        "the output of the original trained network.")

    def test_jit_out_close_to_net_out(self):

        scripted_module = torch.jit.load(self.compiled_path)
        self._test_jit_out_close_to_net_out(scripted_module)

    def test_jit_out_close_to_net_out_without_example_inputs(self):

        scripted_module = jit_script_compile(self.network, example_inputs=None)
        self._test_jit_out_close_to_net_out(scripted_module)


if __name__ == '__main__':
    unittest.main()
