"""Configuration validation should finish before creating an experiment."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from torchbase import TrainingBaseSession
from tests.test_session_resume import ResumeSession


class SessionConfigUnitTest(unittest.TestCase):
    def setUp(self):
        self.config = ResumeSession.get_config()
        self.config["session"]["loss_function_params"] = {"options": ["mean"]}
        self.config["data"] = {"custom": {"values": [1, 2]}}
        self.config["metrics"] = {"custom": ["score"]}
        self.config["network"]["custom"] = {"layers": [2, 1]}

    @staticmethod
    def as_config(parts):
        session, data, metrics, network = parts
        return {"session": session.to_dict(), "data": data, "metrics": metrics, "network": network}

    def test_all_sections_are_copied_including_nested_values(self):
        original = deepcopy(self.config)
        first = self.as_config(TrainingBaseSession.setup_configs(self.config))
        second = self.as_config(TrainingBaseSession.setup_configs(self.config))
        first["data"]["custom"]["values"].clear()
        first["metrics"]["custom"].clear()
        first["network"]["custom"]["layers"].clear()
        self.assertEqual(self.config, original)
        self.assertEqual(second["data"], original["data"])
        self.assertEqual(second["metrics"], original["metrics"])
        self.assertEqual(second["network"], original["network"])
        self.config["data"]["custom"]["values"].append(3)
        self.assertEqual(second["data"], original["data"])

    def test_custom_settings_survive_json_round_trip(self):
        normalized = self.as_config(TrainingBaseSession.setup_configs(self.config))
        loaded = json.loads(json.dumps(normalized))
        self.assertEqual(self.as_config(TrainingBaseSession.setup_configs(loaded)), normalized)

    def test_tuples_follow_json_array_semantics(self):
        self.config["data"]["shape"] = (32, 32)
        normalized = self.as_config(TrainingBaseSession.setup_configs(self.config))
        self.assertEqual(normalized["data"]["shape"], (32, 32))
        loaded = json.loads(json.dumps(normalized))
        self.assertEqual(self.as_config(TrainingBaseSession.setup_configs(loaded))["data"]["shape"], [32, 32])

    def test_missing_and_unknown_sections_are_named(self):
        for key in self.config:
            config = {name: value for name, value in self.config.items() if name != key}
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, key):
                TrainingBaseSession.setup_configs(config)
        for key in ("metrcis", "custom"):
            with self.subTest(unknown=key), self.assertRaisesRegex(ValueError, key):
                TrainingBaseSession.setup_configs({**self.config, key: {}})

    def test_sections_must_be_dictionaries(self):
        for key in self.config:
            for value in (None, [], 1, "config"):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(TypeError, key):
                    TrainingBaseSession.setup_configs({**self.config, key: value})

    def test_config_must_be_a_dictionary(self):
        for value in (None, [], "config", 1):
            with self.subTest(value=value), self.assertRaises(TypeError):
                TrainingBaseSession.setup_configs(value)

    def test_circular_settings_are_rejected_without_mutation(self):
        self.config["data"]["loop"] = self.config["data"]
        with self.assertRaisesRegex(ValueError, "JSON-compatible"):
            TrainingBaseSession.setup_configs(self.config)
        self.assertIs(self.config["data"]["loop"], self.config["data"])
        self.assertNotIn("checkpoint_interval", self.config["session"])

    def test_architecture_is_required_and_must_be_a_nonempty_string(self):
        for network in ({}, {"architecture": None}, {"architecture": 1}, {"architecture": ""}):
            with self.subTest(network=network), self.assertRaisesRegex(ValueError, "architecture"):
                TrainingBaseSession.setup_configs({**self.config, "network": network})

    def test_invalid_config_fails_before_run_directory_setup(self):
        for key in ("session", "data", "metrics", "network"):
            config = {**self.config, key: None}
            with self.subTest(key=key), patch.object(
                    ResumeSession, "setup_run_dir_for_logging",
                    side_effect=AssertionError("Run directory setup should not be reached")) as setup_dir:
                with self.assertRaises(TypeError):
                    ResumeSession(config)
                setup_dir.assert_not_called()

    def test_values_that_cannot_round_trip_are_rejected_before_run_setup(self):
        for value in ({1: "value"}, {"nested": [{None: "value"}]}, {"items": {1, 2}},
                      {"value": object()}, {"value": float("nan")}, {"value": float("inf")}):
            for section in ("data", "metrics", "network", "session"):
                config = deepcopy(self.config)
                if section == "session":
                    config[section]["loss_function_params"] = value
                else:
                    config[section]["custom"] = value
                with self.subTest(section=section, value=value), \
                        patch.object(ResumeSession, "setup_run_dir_for_logging",
                                     side_effect=AssertionError("Run directory setup should not be reached")) as setup_dir:
                    with self.assertRaisesRegex(ValueError, "config|JSON"):
                        ResumeSession(config)
                    setup_dir.assert_not_called()
