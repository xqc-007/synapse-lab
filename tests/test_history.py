"""Tests for experiment recording and export."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from synapse.experiments.history import ExperimentConfig, run_experiment

XOR = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]


class ExperimentHistoryTests(unittest.TestCase):
    def test_records_initial_interval_and_final_loss(self):
        result = run_experiment(
            XOR, ExperimentConfig(epochs=11, record_every=4)
        )
        self.assertEqual([point.epoch for point in result.metrics], [0, 4, 8, 11])
        self.assertEqual(len(result.predictions), len(XOR))

    def test_zero_epochs_records_only_initial_state(self):
        result = run_experiment(XOR, ExperimentConfig(epochs=0))
        self.assertEqual(len(result.metrics), 1)
        self.assertEqual(result.metrics[0].epoch, 0)

    def test_repeated_runs_are_identical(self):
        config = ExperimentConfig(epochs=25, record_every=5)
        first = run_experiment(XOR, config)
        second = run_experiment(XOR, config)
        self.assertEqual(first, second)

    def test_invalid_configuration_and_samples(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(record_every=0)
        with self.assertRaises(ValueError):
            ExperimentConfig(epochs=-1)
        with self.assertRaises(ValueError):
            run_experiment([], ExperimentConfig())
        with self.assertRaises(ValueError):
            run_experiment([([0.0], 1.0)], ExperimentConfig())
        with self.assertRaises(ValueError):
            run_experiment([([0.0, 1.0], 2.0)], ExperimentConfig())

    def test_exports_machine_readable_files(self):
        result = run_experiment(XOR, ExperimentConfig(epochs=5, record_every=2))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "experiment"
            result.save(directory)
            metadata = json.loads((directory / "experiment.json").read_text())
            with (directory / "metrics.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(metadata["config"]["epochs"], 5)
            self.assertEqual(metadata["metric"], "mean_squared_error")
            self.assertEqual([int(row["epoch"]) for row in rows], [0, 2, 4, 5])


if __name__ == "__main__":
    unittest.main()
