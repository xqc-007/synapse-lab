"""Checks for the confusion matrix renderer."""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from synapse.metrics import BinaryMetrics, binary_metrics
from synapse.visualisation.confusion_plot import render_confusion_svg, save_confusion_svg


class ConfusionPlotTests(unittest.TestCase):
    def test_asymmetric_counts_are_in_correct_cells(self):
        metrics = binary_metrics([0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 0, 1])
        root = ET.fromstring(render_confusion_svg(metrics))
        ns = {"s": "http://www.w3.org/2000/svg"}
        cells = [node for node in root.findall(".//s:text", ns)
                 if node.get("font-size") == "24"]
        self.assertEqual([(n.get("x"), n.get("y"), n.text) for n in cells],
                         [("228", "112", "2"), ("408", "112", "1"),
                          ("228", "212", "2"), ("408", "212", "1")])

    def test_single_class(self):
        self.assertIn("Accuracy: 100.0%", render_confusion_svg(BinaryMetrics(0, 4, 0, 0)))

    def test_invalid_counts(self):
        for metrics in (BinaryMetrics(0, 0, 0, 0), BinaryMetrics(-1, 2, 0, 0)):
            with self.assertRaises(ValueError):
                render_confusion_svg(metrics)

    def test_export(self):
        metrics = BinaryMetrics(1, 2, 3, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plots" / "matrix.svg"
            self.assertEqual(save_confusion_svg(metrics, path), path)
            self.assertEqual(path.read_text(encoding="utf-8"), render_confusion_svg(metrics))
