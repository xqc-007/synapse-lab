"""Checks for repeated-seed experiment summaries."""

import math
import unittest

from synapse.experiments.seed_comparison import compare_seeds
from synapse.experiments.xor import SAMPLES


class SeedComparisonTests(unittest.TestCase):
    def test_reports_each_seed_and_summary(self):
        result = compare_seeds(SAMPLES, hidden_size=3, seeds=[2, 4, 6], epochs=10)
        self.assertEqual(result.seeds, (2, 4, 6))
        self.assertEqual(len(result.final_losses), 3)
        self.assertAlmostEqual(result.mean_loss, sum(result.final_losses) / 3)
        self.assertLessEqual(result.minimum_loss, result.mean_loss)
        self.assertGreaterEqual(result.maximum_loss, result.mean_loss)
        self.assertTrue(math.isfinite(result.standard_deviation))

    def test_same_inputs_produce_same_results(self):
        options = dict(hidden_size=4, seeds=[1, 5], epochs=12)
        self.assertEqual(compare_seeds(SAMPLES, **options), compare_seeds(SAMPLES, **options))

    def test_single_seed_has_zero_dispersion(self):
        result = compare_seeds(SAMPLES, hidden_size=3, seeds=[3], epochs=2)
        self.assertEqual(result.standard_deviation, 0.0)
        self.assertEqual(result.minimum_loss, result.maximum_loss)

    def test_rejects_empty_duplicate_or_non_integer_seeds(self):
        for seeds in ([], [1, 1], [True], [1.5], ["7"]):
            with self.subTest(seeds=seeds):
                with self.assertRaises(ValueError):
                    compare_seeds(SAMPLES, hidden_size=3, seeds=seeds, epochs=1)

    def test_rejects_invalid_network_configuration(self):
        with self.assertRaises(ValueError):
            compare_seeds(SAMPLES, hidden_size=0, seeds=[1], epochs=1)


if __name__ == "__main__":
    unittest.main()
