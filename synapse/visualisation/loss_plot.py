"""Render experiment loss curves as standalone SVG files.

The renderer uses the Python standard library and does not require a browser,
plotting package or running web service.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from synapse.experiments.history import Metric


def render_loss_svg(
    metrics: Sequence[Metric], width: int = 720, height: int = 420
) -> str:
    """Return an SVG chart with labelled axes and a linear loss scale."""
    if not metrics:
        raise ValueError("At least one metric is required.")
    if width < 320 or height < 240:
        raise ValueError("Chart dimensions must be at least 320 by 240.")
    if any(
        not isinstance(point.epoch, int)
        or point.epoch < 0
        or not math.isfinite(point.loss)
        or point.loss < 0
        for point in metrics
    ):
        raise ValueError("Epochs and losses must be finite and non-negative.")
    epochs = [point.epoch for point in metrics]
    if any(right <= left for left, right in zip(epochs, epochs[1:])):
        raise ValueError("Epochs must be strictly increasing.")

    left, right, top, bottom = 86, width - 30, 45, height - 75
    x_min, x_max = epochs[0], epochs[-1]
    y_max = max(point.loss for point in metrics)
    y_max = y_max * 1.05 if y_max > 0 else 1.0

    def x(epoch: int) -> float:
        fraction = (epoch - x_min) / (x_max - x_min) if x_max != x_min else 0.5
        return left + fraction * (right - left)

    def y(loss: float) -> float:
        return bottom - (loss / y_max) * (bottom - top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        'role="img" aria-labelledby="chart-title chart-desc">',
        '<title id="chart-title">Training loss</title>',
        '<desc id="chart-desc">Mean squared error by training epoch, '
        'using a linear vertical scale.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    for tick in range(5):
        value = y_max * tick / 4
        position = y(value)
        parts.extend([
            f'<line x1="{left}" y1="{position:.2f}" x2="{right}" '
            f'y2="{position:.2f}" stroke="#e5e7eb"/>',
            f'<text x="{left - 12}" y="{position + 4:.2f}" '
            f'text-anchor="end" font-size="12" fill="#4b5563">'
            f'{value:.3g}</text>',
        ])
    for tick in range(5):
        value = x_min + (x_max - x_min) * tick / 4
        position = left + (right - left) * tick / 4
        parts.append(
            f'<text x="{position:.2f}" y="{bottom + 23}" '
            f'text-anchor="middle" font-size="12" fill="#4b5563">'
            f'{value:.0f}</text>'
        )
    points = " ".join(f"{x(p.epoch):.2f},{y(p.loss):.2f}" for p in metrics)
    parts.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" '
        'stroke="#374151"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" '
        'stroke="#374151"/>',
        f'<polyline points="{points}" fill="none" stroke="#2563eb" '
        'stroke-width="2" stroke-linejoin="round"/>',
        f'<text x="{(left + right) / 2:.1f}" y="{height - 24}" '
        'text-anchor="middle" font-size="14" fill="#111827">'
        'Training epoch</text>',
        f'<text x="22" y="{(top + bottom) / 2:.1f}" '
        'transform="rotate(-90 22 '
        f'{(top + bottom) / 2:.1f})" '
        'text-anchor="middle" font-size="14" fill="#111827">'
        'Mean squared error</text>',
        '</svg>',
    ])
    return "\n".join(parts) + "\n"


def save_loss_svg(
    metrics: Sequence[Metric], destination: str | Path
) -> Path:
    """Write a standalone loss plot and return its path."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_loss_svg(metrics), encoding="utf-8")
    return path
