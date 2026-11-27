"""Export in-sample classification diagnostics for a recorded experiment."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from synapse.experiments.history import ExperimentResult
from synapse.metrics import binary_metrics
from synapse.visualisation.confusion_plot import render_confusion_svg


def save_classification_report(
    result: ExperimentResult, targets: Sequence[int],
    directory: str | Path, threshold: float = 0.5,
) -> Path:
    """Export experiment history and diagnostics for aligned training labels.

    These predictions describe the training samples, not held-out performance.
    """
    metrics = binary_metrics(targets, result.predictions, threshold)
    svg = render_confusion_svg(metrics)
    report = {
        "evaluation": "training_samples",
        "threshold": threshold,
        "sample_count": len(targets),
        "counts": asdict(metrics),
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
    }
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    destination = Path(directory)
    result.save(destination)
    (destination / "confusion.svg").write_text(svg, encoding="utf-8")
    path = destination / "classification.json"
    path.write_text(payload, encoding="utf-8")
    return path
