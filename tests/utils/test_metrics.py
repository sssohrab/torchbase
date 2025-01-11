import unittest
from torchbase.utils.metrics import BaseMetricsClass
import inspect


class DummyMetrics(BaseMetricsClass):
    @staticmethod
    def metric1(*, x, y) -> float:
        return x + y

    @staticmethod
    def metric2(*, x, y) -> float:
        return x * y

    @staticmethod
    def _private_method(data) -> None:
        return None


class BaseMetricsClassUnitTest(unittest.TestCase):
    def setUp(self):
        self.metrics_instance = DummyMetrics(keyword_maps=None)

    def test_get_all_metric_functionals(self):
        functionals = self.metrics_instance.get_all_metric_functionals_dict()
        self.assertIn('metric1', functionals)
        self.assertIn('metric2', functionals)
        self.assertNotIn('_private_method', functionals)
        for name, func in functionals.items():
            with self.assertRaises(TypeError):
                func(1, 2)  # Calling without keyword should fail

    def test_get_metrics_invalid(self):
        with self.assertRaises(ValueError):
            self.metrics_instance.get_metrics(['non_existent_metric'])

    def test_get_metrics_valid(self):
        metrics_dict = self.metrics_instance.get_metrics(['metric1', 'metric2'])
        for name, func in metrics_dict.items():
            func(x=1, y=2)
        self.assertTrue(callable(metrics_dict['metric1']))
        self.assertTrue(callable(metrics_dict['metric2']))

        self.assertEqual(metrics_dict["metric1"](x=3, y=7), 10)
        self.assertEqual(metrics_dict["metric2"](x=3, y=7), 21)

        self.assertIn("x", inspect.signature(metrics_dict["metric1"]).parameters.keys())
        self.assertIn("y", inspect.signature(metrics_dict["metric1"]).parameters.keys())

    def test_get_metrics_with_mapped_keys(self):
        keyword_maps = {"a": "x", "b": "y"}
        metrics_instance = DummyMetrics(keyword_maps=keyword_maps)
        metrics_to_get = ["metric1", "metric2"]

        metrics_dict = metrics_instance.get_metrics(metrics_to_get)

        self.assertTrue(callable(metrics_dict['metric1']))
        self.assertTrue(callable(metrics_dict['metric2']))

        self.assertEqual(metrics_dict["metric1"](a=3, b=7), 10)
        self.assertEqual(metrics_dict["metric2"](a=3, b=7), 21)

        self.assertIn("a", inspect.signature(metrics_dict["metric1"]).parameters.keys())
        self.assertIn("b", inspect.signature(metrics_dict["metric1"]).parameters.keys())


if __name__ == '__main__':
    unittest.main()
