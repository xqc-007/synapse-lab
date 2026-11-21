"""Tests for comparing training runs on shared axes."""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from synapse.experiments.history import Metric
from synapse.visualisation.comparison_plot import (
    render_comparison_svg,
    save_comparison_svg,
)


class ComparisonPlotTests(unittest.TestCase):
    def test_renders_two_runs_as_valid_svg(self):
        runs = {
            "small network": [Metric(0, 0.4), Metric(10, 0.2)],
            "large network": [Metric(0, 0.5), Metric(5, 0.3), Metric(20, 0.1)],
        }
        svg = render_comparison_svg(runs)
        root = ET.fromstring(svg)
        self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
        self.assertEqual(svg.count("<polyline"), 2)
        self.assertIn("large network", svg)
        self.assertIn("20", svg)

    def test_escapes_experiment_names(self):
        svg = render_comparison_svg({"run <A> & B": [Metric(0, 0.0)]})
        ET.fromstring(svg)
        self.assertIn("run &lt;A&gt; &amp; B", svg)

    def test_rejects_invalid_series(self):
        invalid = (
            {},
            {"empty": []},
            {"reverse": [Metric(3, 0.2), Metric(2, 0.1)]},
            {"duplicate": [Metric(1, 0.2), Metric(1, 0.1)]},
            {"negative": [Metric(0, -0.1)]},
            {"blank": [Metric(0, float("nan"))]},
            {"": [Metric(0, 0.2)]},
        )
        for runs in invalid:
            with self.subTest(runs=runs), self.assertRaises(ValueError):
                render_comparison_svg(runs)

    def test_supports_zero_loss_and_single_epoch(self):
        svg = render_comparison_svg({"baseline": [Metric(0, 0.0)]})
        ET.fromstring(svg)
        self.assertNotIn("nan", svg.lower())

    def test_rejects_excessive_runs_and_small_chart(self):
        runs = {str(i): [Metric(0, 0.1)] for i in range(7)}
        with self.assertRaises(ValueError):
            render_comparison_svg(runs)
        with self.assertRaises(ValueError):
            render_comparison_svg({"one": [Metric(0, 0.1)]}, width=200)

    def test_writes_svg(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plots" / "comparison.svg"
            saved = save_comparison_svg({"one": [Metric(0, 0.2)]}, path)
            self.assertEqual(saved, path)
            ET.parse(path)


if __name__ == "__main__":
    unittest.main()
