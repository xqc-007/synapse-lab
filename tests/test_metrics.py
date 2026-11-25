"""Checks for binary classification diagnostics."""

import unittest

from synapse.metrics import BinaryMetrics, binary_metrics


class BinaryMetricsTests(unittest.TestCase):
    def test_all_four_outcomes(self):
        result = binary_metrics([1, 0, 0, 1], [0.9, 0.1, 0.8, 0.2])
        self.assertEqual(result, BinaryMetrics(1, 1, 1, 1))
        self.assertEqual(result.accuracy, 0.5)
        self.assertEqual(result.precision, 0.5)
        self.assertEqual(result.recall, 0.5)

    def test_threshold_is_inclusive(self):
        self.assertEqual(binary_metrics([1], [0.7], 0.7).accuracy, 1)

    def test_undefined_ratios(self):
        result = binary_metrics([0], [0.1])
        self.assertIsNone(result.precision)
        self.assertIsNone(result.recall)
        self.assertEqual(result.accuracy, 1)

    def test_invalid_inputs(self):
        for targets, probabilities, threshold in [
            ([], [], 0.5), ([1], [], 0.5), ([0.5], [0.5], 0.5),
            ([1], [float('nan')], 0.5), ([1], [1.1], 0.5),
            ([1], [0.5], -0.1),
        ]:
            with self.subTest(targets=targets, probabilities=probabilities):
                with self.assertRaises(ValueError):
                    binary_metrics(targets, probabilities, threshold)
