import json
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from synapse.experiments.classification_report import save_classification_report
from synapse.experiments.history import ExperimentConfig, ExperimentResult, Metric


class ClassificationReportTests(unittest.TestCase):
    def setUp(self):
        self.result = ExperimentResult(
            ExperimentConfig(epochs=0), (Metric(0, 0.25),),
            (0.1, 0.8, 0.6, 0.2),
        )

    def test_report_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_classification_report(self.result, [0, 1, 0, 1], directory)
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(report["evaluation"], "training_samples")
            self.assertEqual(report["accuracy"], 0.5)
            self.assertEqual(report["counts"]["false_positive"], 1)
            self.assertEqual(report["counts"]["false_negative"], 1)
            ElementTree.parse(Path(directory) / "confusion.svg")
            self.assertTrue((Path(directory) / "metrics.csv").is_file())
            self.assertTrue((Path(directory) / "experiment.json").is_file())

    def test_invalid_labels_do_not_create_report(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "report"
            with self.assertRaises(ValueError):
                save_classification_report(self.result, [0, 1], destination)
            self.assertFalse(destination.exists())

    def test_threshold_and_undefined_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_classification_report(self.result, [0, 1, 0, 1],
                                              directory, threshold=1.0)
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsNone(report["precision"])
            self.assertEqual(report["recall"], 0.0)
