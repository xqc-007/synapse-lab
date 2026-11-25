"""Binary classification diagnostics for probability predictions."""

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BinaryMetrics:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    @property
    def accuracy(self) -> float:
        total = (self.true_positive + self.true_negative
                 + self.false_positive + self.false_negative)
        return (self.true_positive + self.true_negative) / total

    @property
    def precision(self) -> float | None:
        total = self.true_positive + self.false_positive
        return self.true_positive / total if total else None

    @property
    def recall(self) -> float | None:
        total = self.true_positive + self.false_negative
        return self.true_positive / total if total else None


def binary_metrics(targets: Sequence[int], probabilities: Sequence[float],
                   threshold: float = 0.5) -> BinaryMetrics:
    """Count outcomes; probabilities at the threshold are positive.

    Undefined precision or recall is represented by None, not zero.
    Targets must be hard binary labels rather than soft training targets.
    """
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and between 0 and 1.")
    if len(targets) == 0 or len(targets) != len(probabilities):
        raise ValueError("Provide equally sized, non-empty sequences.")
    tp = tn = fp = fn = 0
    for target, probability in zip(targets, probabilities):
        if target not in (0, 1):
            raise ValueError("Targets must be binary labels.")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Probabilities must be finite and between 0 and 1.")
        positive = probability >= threshold
        if target == 1:
            tp += int(positive)
            fn += int(not positive)
        else:
            fp += int(positive)
            tn += int(not positive)
    return BinaryMetrics(tp, tn, fp, fn)
