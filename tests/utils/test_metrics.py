import unittest
from torchbase.utils.metrics import BaseMetricsClass
from torchbase.utils.logger import LoggableParams
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


class OptionalMetrics(BaseMetricsClass):
    @staticmethod
    def scaled(*, x: float, y: float, scale: float = 2.0) -> float:
        """A metric with a default that should survive argument mapping."""
        return (x + y) * scale

    @staticmethod
    def single(*, y: float) -> float:
        return y

    @staticmethod
    def with_extras(*, x: float, scale: float = 2.0, **extras) -> float:
        return x * scale + extras.get("bonus", 0.0)


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


    def test_partial_mapping_preserves_unmapped_arguments_in_logger(self):
        functions = DummyMetrics({"a": "x"}).get_metrics(["metric1", "metric2"])
        for function in functions.values():
            self.assertEqual(list(inspect.signature(function).parameters), ["a", "y"])
        self.assertEqual(LoggableParams(functions)(a=3.0, y=7.0, unused=99.0),
                         {"metric1": 10.0, "metric2": 21.0})

    def test_defaults_annotations_and_metadata_are_preserved(self):
        function = OptionalMetrics({"a": "x"}).get_metrics(["scaled"])["scaled"]
        original = inspect.signature(OptionalMetrics.scaled)
        expected = original.replace(parameters=[parameter.replace(name="a") if name == "x" else parameter
                                                for name, parameter in original.parameters.items()])
        self.assertEqual(inspect.signature(function), expected)
        self.assertEqual(function.__name__, OptionalMetrics.scaled.__name__)
        self.assertEqual(function.__doc__, OptionalMetrics.scaled.__doc__)
        logger = LoggableParams({"scaled": function})
        self.assertEqual(logger(a=1.0, y=2.0), {"scaled": 6.0})
        self.assertEqual(logger(a=1.0, y=2.0, scale=3.0), {"scaled": 9.0})
        with self.assertRaises(TypeError):
            logger(a=1.0)

    def test_mapped_optional_argument_keeps_its_default(self):
        function = OptionalMetrics({"a": "x", "factor": "scale"}).get_metrics(["scaled"])["scaled"]
        self.assertEqual(inspect.signature(function).parameters["factor"].default, 2.0)
        logger = LoggableParams({"scaled": function})
        self.assertEqual(logger(a=1.0, y=2.0), {"scaled": 6.0})
        self.assertEqual(logger(a=1.0, y=2.0, factor=4.0), {"scaled": 12.0})

    def test_shared_mapping_only_exposes_arguments_used_by_each_metric(self):
        functions = OptionalMetrics({"a": "x", "b": "y", "factor": "scale"}).get_metrics(["scaled", "single"])
        self.assertEqual(list(inspect.signature(functions["scaled"]).parameters), ["a", "b", "factor"])
        self.assertEqual(list(inspect.signature(functions["single"]).parameters), ["b"])
        self.assertEqual(LoggableParams(functions)(a=1.0, b=2.0, factor=4.0), {"scaled": 12.0, "single": 2.0})

    def test_irrelevant_mapping_leaves_metric_unchanged(self):
        function = OptionalMetrics({"a": "x"}).get_metrics(["single"])["single"]
        self.assertEqual(inspect.signature(function), inspect.signature(OptionalMetrics.single))
        self.assertEqual(LoggableParams({"single": function})(a=1.0, y=2.0), {"single": 2.0})

    def test_duplicate_mapping_targets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same metric argument"):
            DummyMetrics({"a": "x", "b": "x"}).get_metrics(["metric1"])

    def test_invalid_mapping_types_are_rejected(self):
        for mapping in ([], "a", {1: "x"}, {"a": 1}):
            with self.subTest(mapping=mapping), self.assertRaises(TypeError):
                DummyMetrics(mapping)

    def test_invalid_aliases_and_var_keyword_name_collisions_are_rejected(self):
        for source in ("not-an-identifier", "class", "extras"):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "conflict.*with_extras"):
                OptionalMetrics({source: "x"}).get_metrics(["with_extras"])

    def test_mapping_cannot_hide_an_unmapped_parameter(self):
        for mapping in ({"y": "x"}, {"scale": "x"}):
            with self.subTest(mapping=mapping), self.assertRaisesRegex(ValueError, "conflict"):
                OptionalMetrics(mapping).get_metrics(["scaled"])

    def test_alias_and_original_cannot_silently_overwrite_each_other(self):
        function = DummyMetrics({"a": "x"}).get_metrics(["metric1"])["metric1"]
        for values in ({"a": 1.0, "x": 2.0, "y": 3.0}, {"x": 2.0, "a": 1.0, "y": 3.0}):
            with self.subTest(values=values), self.assertRaisesRegex(TypeError, "multiple values.*x"):
                function(**values)

    def test_identity_chained_and_swapped_mappings(self):
        for mapping, values in (({"x": "x"}, {"x": 1.0, "y": 2.0}),
                                ({"a": "x", "x": "y"}, {"a": 1.0, "x": 2.0}),
                                ({"y": "x", "x": "y"}, {"y": 1.0, "x": 2.0})):
            with self.subTest(mapping=mapping):
                function = DummyMetrics(mapping).get_metrics(["metric1"])["metric1"]
                self.assertEqual(LoggableParams({"value": function})(**values), {"value": 3.0})

    def test_mapping_is_copied_and_generated_functions_remain_consistent(self):
        mapping = {"a": "x"}
        metrics = DummyMetrics(mapping)
        mapping["a"] = "y"
        function = metrics.get_metrics(["metric1"])["metric1"]
        metrics.keyword_maps["a"] = "y"
        self.assertEqual(list(inspect.signature(function).parameters), ["a", "y"])
        self.assertEqual(function(a=1.0, y=2.0), 3.0)

    def test_var_keyword_parameter_is_preserved(self):
        function = OptionalMetrics({"a": "x"}).get_metrics(["with_extras"])["with_extras"]
        params = inspect.signature(function).parameters
        self.assertEqual(list(params), ["a", "scale", "extras"])
        self.assertEqual(params["extras"].kind, inspect.Parameter.VAR_KEYWORD)
        self.assertEqual(function(a=2.0, bonus=3.0), 7.0)

    def test_mapped_extra_keywords_remain_visible_to_logger(self):
        function = OptionalMetrics({"a": "x", "b": "bonus"}).get_metrics(["with_extras"])["with_extras"]
        self.assertEqual(LoggableParams({"value": function})(a=2.0, b=3.0, scale=4.0), {"value": 11.0})
        self.assertEqual(inspect.signature(function).parameters["extras"].kind, inspect.Parameter.VAR_KEYWORD)

    def test_empty_mapping_preserves_original_function(self):
        for mapping in (None, {}):
            function = DummyMetrics(mapping).get_metrics(["metric1"])["metric1"]
            self.assertIs(function, DummyMetrics.metric1)


if __name__ == '__main__':
    unittest.main()
