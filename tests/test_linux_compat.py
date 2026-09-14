"""Small cross-platform checks that do not require a training run."""

import json
import platform
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class CrossPlatformConfigurationTest(unittest.TestCase):
    def test_source_data_and_entrypoints_are_relative_to_repository(self):
        from experiment_core import CODE_DIR, DATA_DIR

        self.assertEqual(CODE_DIR, ROOT)
        self.assertTrue((DATA_DIR / "0060merge_open_close_final.csv").is_file())
        self.assertTrue((DATA_DIR / "3068merge_open_close_final.csv").is_file())
        for name in ("train.py", "main.py", "Fig_main.py", "Appendix_Fig_main.py"):
            self.assertTrue((ROOT / name).is_file(), name)

    def test_shared_esg_thresholds_are_from_raw_cross_section(self):
        from experiment_core import esg_thresholds_common

        thresholds = esg_thresholds_common()
        self.assertEqual(set(thresholds), {"25%", "50%"})
        self.assertLessEqual(thresholds["25%"], thresholds["50%"])
        self.assertAlmostEqual(thresholds["25%"], 5.52, places=2)
        self.assertAlmostEqual(thresholds["50%"], 6.02, places=2)

    def test_paper_locked_calibration_keeps_reported_parameters(self):
        from experiment_core import (
            PAPER_HYPERPARAMETERS,
            lambda_mart_objective,
        )

        # T4M10.py/T4C10.py use pairwise LambdaMART with NDCG as the metric.
        self.assertEqual(lambda_mart_objective("0060"), "rank:map")
        self.assertEqual(lambda_mart_objective("3068"), "rank:ndcg")
        self.assertEqual(
            PAPER_HYPERPARAMETERS["LambdaMART"]["0060"],
            {"n_estimators": 1000, "max_depth": 5, "learning_rate": 0.001},
        )
        self.assertEqual(
            PAPER_HYPERPARAMETERS["LambdaMART"]["3068"],
            {"n_estimators": 1000, "max_depth": 6, "learning_rate": 0.1},
        )
        self.assertEqual(PAPER_HYPERPARAMETERS["LTR-DQN"]["learning_rate"], 0.002)

    def test_host_automatically_selects_its_parameter_file(self):
        from runtime_config import (
            ACTIVE_PARAMETER_FILE,
            ACTIVE_PLATFORM_PROFILE,
            parameter_file_for_system,
        )

        expected_profiles = {
            "Windows": "windows-reference",
            "Linux": "linux-emergency-compatibility",
        }
        detected = platform.system()
        self.assertIn(detected, expected_profiles)
        self.assertEqual(ACTIVE_PARAMETER_FILE, parameter_file_for_system(detected))
        self.assertEqual(ACTIVE_PLATFORM_PROFILE, expected_profiles[detected])

    def test_windows_and_linux_parameter_profiles_are_valid_and_distinct(self):
        from runtime_config import load_platform_parameters, parameter_file_for_system

        windows = load_platform_parameters("Windows")
        linux = load_platform_parameters("Linux")
        self.assertEqual(parameter_file_for_system("Windows").name, "parameters_windows.txt")
        self.assertEqual(parameter_file_for_system("Linux").name, "parameters_linux.txt")
        self.assertEqual(windows["profile"], "windows-reference")
        self.assertEqual(linux["profile"], "linux-emergency-compatibility")
        self.assertEqual(windows["stage_seeds"]["0060"]["4"]["evaluation"], 59)
        self.assertEqual(linux["stage_seeds"]["0060"]["4"]["evaluation"], 5)
        self.assertNotEqual(windows["stage_seeds"], linux["stage_seeds"])
        for config in (windows, linux):
            # Paper-reported learning rates remain in experiment_core.py and
            # cannot diverge between the two platform profiles.
            self.assertNotIn("learning_rate", json.dumps(config))

        with patch("runtime_config.platform.system", return_value="Windows"):
            self.assertEqual(load_platform_parameters()["profile"], "windows-reference")
        with patch("runtime_config.platform.system", return_value="Linux"):
            self.assertEqual(
                load_platform_parameters()["profile"],
                "linux-emergency-compatibility",
            )

    def test_unsupported_operating_system_fails_closed(self):
        from runtime_config import parameter_file_for_system

        with self.assertRaisesRegex(RuntimeError, "Unsupported operating system"):
            parameter_file_for_system("Darwin")

    def test_github_actions_are_windows_only(self):
        workflow_dir = ROOT / ".github" / "workflows"
        workflows = sorted(
            [*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml")]
        )
        self.assertEqual(
            [path.name for path in workflows],
            ["reproduce-core.yml", "reproduce-t6.yml"],
        )
        for path in workflows:
            content = path.read_text(encoding="utf-8")
            self.assertIn("runs-on: windows-2022", content, path.name)
            self.assertNotIn("ubuntu-", content, path.name)
            self.assertIn("windows-reference", content, path.name)

    def test_linux_local_support_files_exist(self):
        for name in (
            "requirements-linux.txt",
            "environment-linux.yml",
            "run_linux.sh",
            "run_t6_linux.sh",
            "parameters_linux.txt",
            "parameters_windows.txt",
        ):
            self.assertTrue((ROOT / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
