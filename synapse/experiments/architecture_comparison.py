"""Compare network widths using identical initialisation seeds and budgets."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence

from synapse.experiments.history import Sample
from synapse.experiments.seed_comparison import SeedComparison, compare_seeds


@dataclass(frozen=True)
class ArchitectureComparison:
    """Paired final-loss comparison across two hidden-layer widths."""

    baseline: SeedComparison
    candidate: SeedComparison
    seeds: tuple[int, ...]
    loss_differences: tuple[float, ...]
    mean_difference: float
    median_difference: float
    candidate_wins: int
    ties: int
    baseline_wins: int


def compare_architectures(
    samples: Sequence[Sample],
    *,
    baseline_hidden_size: int,
    candidate_hidden_size: int,
    seeds: Sequence[int],
    epochs: int = 2000,
    learning_rate: float = 0.8,
    input_size: int = 2,
    record_every: int = 100,
    tie_tolerance: float = 1e-12,
) -> ArchitectureComparison:
    """Compare final MSE for each matching seed.

    Differences are candidate minus baseline: negative values favour the
    candidate. This is a descriptive paired comparison, not a significance
    test. Equal seeds control the seed labels, not identical initial weights
    across networks of different widths.
    """
    if baseline_hidden_size == candidate_hidden_size:
        raise ValueError("Choose two different hidden-layer widths.")
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("Tie tolerance must be finite and non-negative.")

    selected = tuple(seeds)
    options = dict(
        samples=samples,
        seeds=selected,
        epochs=epochs,
        learning_rate=learning_rate,
        input_size=input_size,
        record_every=record_every,
    )
    baseline = compare_seeds(hidden_size=baseline_hidden_size, **options)
    candidate = compare_seeds(hidden_size=candidate_hidden_size, **options)
    differences = tuple(
        candidate_loss - baseline_loss
        for baseline_loss, candidate_loss in zip(
            baseline.final_losses, candidate.final_losses
        )
    )
    wins = sum(delta < -tie_tolerance for delta in differences)
    losses = sum(delta > tie_tolerance for delta in differences)
    return ArchitectureComparison(
        baseline=baseline,
        candidate=candidate,
        seeds=selected,
        loss_differences=differences,
        mean_difference=statistics.fmean(differences),
        median_difference=statistics.median(differences),
        candidate_wins=wins,
        ties=len(differences) - wins - losses,
        baseline_wins=losses,
    )
