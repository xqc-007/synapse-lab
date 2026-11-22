"""Summarise repeated experiments without assuming one seed is representative."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence

from synapse.experiments.history import ExperimentConfig, Sample, run_experiment


@dataclass(frozen=True)
class SeedComparison:
    """Final-loss distribution for one model configuration across seeds."""

    hidden_size: int
    seeds: tuple[int, ...]
    final_losses: tuple[float, ...]
    mean_loss: float
    standard_deviation: float
    minimum_loss: float
    maximum_loss: float


def compare_seeds(
    samples: Sequence[Sample],
    *,
    hidden_size: int,
    seeds: Sequence[int],
    epochs: int = 2000,
    learning_rate: float = 0.8,
    input_size: int = 2,
    record_every: int = 100,
) -> SeedComparison:
    """Train independent initialisations and summarise their final MSE.

    Standard deviation is the population standard deviation of the supplied
    runs, not an estimate of uncertainty or a statistical significance test.
    """
    selected = tuple(seeds)
    if not selected:
        raise ValueError("At least one seed is required.")
    if any(type(seed) is not int for seed in selected):
        raise ValueError("Seeds must be integers.")
    if len(set(selected)) != len(selected):
        raise ValueError("Seeds must be distinct.")

    losses = []
    for seed in selected:
        config = ExperimentConfig(
            input_size=input_size,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            seed=seed,
            epochs=epochs,
            record_every=record_every,
        )
        result = run_experiment(samples, config)
        losses.append(result.metrics[-1].loss)

    return SeedComparison(
        hidden_size=hidden_size,
        seeds=selected,
        final_losses=tuple(losses),
        mean_loss=statistics.fmean(losses),
        standard_deviation=statistics.pstdev(losses),
        minimum_loss=min(losses),
        maximum_loss=max(losses),
    )
