"""Record reproducible training metrics without external dependencies."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from synapse.network import DenseNetwork

Sample = tuple[Sequence[float], float]


@dataclass(frozen=True)
class ExperimentConfig:
    input_size: int = 2
    hidden_size: int = 6
    learning_rate: float = 0.8
    seed: int = 7
    epochs: int = 6000
    record_every: int = 100

    def __post_init__(self) -> None:
        if self.input_size <= 0 or self.hidden_size <= 0:
            raise ValueError("Layer sizes must be positive.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("The learning rate must be finite and positive.")
        if self.epochs < 0 or self.record_every <= 0:
            raise ValueError("Epochs must be non-negative and record_every positive.")


@dataclass(frozen=True)
class Metric:
    epoch: int
    loss: float


@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    metrics: tuple[Metric, ...]
    predictions: tuple[float, ...]
    evaluation_predictions: tuple[float, ...] | None = None
    evaluation_loss: float | None = None

    def save(self, directory: str | Path) -> None:
        """Write metadata and a chart-ready CSV to the chosen directory."""
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        metadata = {
            "config": asdict(self.config),
            "predictions": list(self.predictions),
            "metric": "mean_squared_error",
        }
        if self.evaluation_predictions is not None:
            metadata["evaluation_predictions"] = list(self.evaluation_predictions)
            metadata["evaluation_loss"] = self.evaluation_loss
        (destination / "experiment.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        with (destination / "metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(("epoch", "loss"))
            writer.writerows((item.epoch, item.loss) for item in self.metrics)


def run_experiment(
    samples: Sequence[Sample], config: ExperimentConfig,
    *, evaluation_samples: Sequence[Sample] | None = None,
) -> ExperimentResult:
    """Train the network and record the initial and sampled losses."""
    if not samples:
        raise ValueError("The dataset cannot be empty.")

    # Validate the entire dataset before any training takes place.
    if evaluation_samples is not None and not evaluation_samples:
        raise ValueError("The evaluation dataset cannot be empty.")
    for inputs, target in (*samples, *(evaluation_samples or ())):
        if len(inputs) != config.input_size:
            raise ValueError("Sample dimensions do not match the configuration.")
        if not all(math.isfinite(value) for value in inputs):
            raise ValueError("Inputs must be finite.")
        if not math.isfinite(target) or not 0 <= target <= 1:
            raise ValueError("Targets must be between 0 and 1.")

    network = DenseNetwork(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        learning_rate=config.learning_rate,
        seed=config.seed,
    )
    metrics = [Metric(epoch=0, loss=network.loss(samples))]
    for epoch in range(1, config.epochs + 1):
        loss = network.train_step(samples)
        if epoch % config.record_every == 0 or epoch == config.epochs:
            metrics.append(Metric(epoch=epoch, loss=loss))

    return ExperimentResult(
        config=config,
        metrics=tuple(metrics),
        predictions=tuple(network.predict(inputs) for inputs, _ in samples),
        evaluation_predictions=(None if evaluation_samples is None else
                                tuple(network.predict(inputs) for inputs, _ in evaluation_samples)),
        evaluation_loss=(None if evaluation_samples is None else network.loss(evaluation_samples)),
    )
