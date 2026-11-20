"""Checks for standalone SVG loss charts."""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from synapse.experiments.history import Metric
from synapse.visualisation.loss_plot import render_loss_svg, save_loss_svg


class LossPlotTests(unittest.TestCase):
    def test_renders_accessible_svg(self):
        svg = render_loss_svg([Metric(0, 0.25), Metric(10, 0.1)])
        root = ET.fromstring(svg)
        self.assertTrue(root.tag.endswith("svg"))
        self.assertIn("Training epoch", svg)
        self.assertIn("Mean squared error", svg)
        self.assertIn("0.25", svg)

    def test_single_point_and_zero_loss(self):
        root = ET.fromstring(render_loss_svg([Metric(0, 0.0)]))
        self.assertTrue(root.tag.endswith("svg"))

    def test_rejects_invalid_data(self):
        for points in (
            [],
            [Metric(1, -0.1)],
            [Metric(0, float("nan"))],
            [Metric(3, 0.2), Metric(2, 0.1)],
            [Metric(2, 0.2), Metric(2, 0.1)],
        ):
            with self.subTest(points=points):
                with self.assertRaises(ValueError):
                    render_loss_svg(points)
        with self.assertRaises(ValueError):
            render_loss_svg([Metric(0, 0.1)], width=100)

    def test_writes_svg(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "plots" / "loss.svg"
            saved = save_loss_svg([Metric(0, 0.2)], destination)
            self.assertEqual(saved, destination)
            self.assertTrue(saved.is_file())
            ET.parse(saved)


if __name__ == "__main__":
    unittest.main()
