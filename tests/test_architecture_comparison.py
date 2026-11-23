"""Paired architecture comparison behaviour."""

import math
import unittest

from synapse.experiments.architecture_comparison import compare_architectures
from synapse.experiments.xor import SAMPLES


class ArchitectureComparisonTests(unittest.TestCase):
    def test_paired_differences_match_individual_losses(self):
        result = compare_architectures(
            SAMPLES, baseline_hidden_size=3, candidate_hidden_size=5,
            seeds=[7, 2, 11], epochs=8
        )
        self.assertEqual(result.seeds, (7, 2, 11))
        self.assertEqual(len(result.loss_differences), 3)
        for difference, baseline, candidate in zip(
            result.loss_differences,
            result.baseline.final_losses,
            result.candidate.final_losses,
        ):
            self.assertAlmostEqual(difference, candidate - baseline)
        self.assertEqual(
            result.candidate_wins + result.ties + result.baseline_wins, 3
        )
        self.assertAlmostEqual(
            result.mean_difference, sum(result.loss_differences) / 3
        )

    def test_reproducible(self):
        options = dict(
            baseline_hidden_size=2, candidate_hidden_size=4,
            seeds=[1, 3], epochs=4
        )
        self.assertEqual(
            compare_architectures(SAMPLES, **options),
            compare_architectures(SAMPLES, **options),
        )

    def test_large_tolerance_counts_ties(self):
        result = compare_architectures(
            SAMPLES, baseline_hidden_size=2, candidate_hidden_size=3,
            seeds=[1, 2], epochs=1, tie_tolerance=1
        )
        self.assertEqual(result.ties, 2)

    def test_rejects_identical_widths(self):
        with self.assertRaises(ValueError):
            compare_architectures(
                SAMPLES, baseline_hidden_size=3, candidate_hidden_size=3,
                seeds=[1], epochs=1
            )

    def test_rejects_invalid_tolerance(self):
        for tolerance in (-1, math.nan, math.inf):
            with self.subTest(tolerance=tolerance):
                with self.assertRaises(ValueError):
                    compare_architectures(
                        SAMPLES, baseline_hidden_size=2,
                        candidate_hidden_size=3, seeds=[1],
                        epochs=1, tie_tolerance=tolerance
                    )

    def test_rejects_invalid_seeds(self):
        for seeds in ([], [1, 1], [True]):
            with self.subTest(seeds=seeds):
                with self.assertRaises(ValueError):
                    compare_architectures(
                        SAMPLES, baseline_hidden_size=2,
                        candidate_hidden_size=3, seeds=seeds, epochs=1
                    )


if __name__ == "__main__":
    unittest.main()
