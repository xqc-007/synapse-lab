"""Compare recorded training losses using a restrained standalone SVG chart."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence
from xml.sax.saxutils import escape

from synapse.experiments.history import Metric

_PALETTE = ("#2563eb", "#b45309", "#047857", "#7c3aed", "#be123c", "#475569")


def render_comparison_svg(
    experiments: Mapping[str, Sequence[Metric]],
    width: int = 800,
    height: int = 460,
) -> str:
    """Render loss curves against shared epoch and loss axes.

    The loss axis starts at zero. Epochs use their actual numerical positions,
    so runs with different recording intervals remain comparable.
    """
    if not experiments:
        raise ValueError("At least one experiment is required.")
    if len(experiments) > len(_PALETTE):
        raise ValueError("At most six experiments can be compared.")
    if width < 480 or height < 300:
        raise ValueError("Chart dimensions must be at least 480 by 300.")

    for name, metrics in experiments.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Experiment names must be non-empty strings.")
        if not metrics:
            raise ValueError("Every experiment must contain at least one metric.")
        previous = -1
        for point in metrics:
            if (
                type(point.epoch) is not int
                or point.epoch < 0
                or point.epoch <= previous
                or not math.isfinite(point.loss)
                or point.loss < 0
            ):
                raise ValueError("Epochs must increase and losses must be finite and non-negative.")
            previous = point.epoch

    left, right, top, bottom = 90, width - 40, 45, height - 115
    max_epoch = max(point.epoch for metrics in experiments.values() for point in metrics)
    max_loss = max(point.loss for metrics in experiments.values() for point in metrics)
    y_limit = max_loss * 1.05 if max_loss else 1.0

    def x(epoch: int) -> float:
        return left + (epoch / max_epoch if max_epoch else 0.5) * (right - left)

    def y(loss: float) -> float:
        return bottom - loss / y_limit * (bottom - top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Experiment loss comparison</title>',
        '<desc id="desc">Mean squared error by training epoch for multiple experiments.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    for tick in range(5):
        loss = y_limit * tick / 4
        yy = y(loss)
        parts.append(
            f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" '
            'stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{yy + 4:.2f}" text-anchor="end" '
            f'font-size="12" fill="#4b5563">{loss:.3g}</text>'
        )
        epoch = max_epoch * tick / 4
        xx = left + (right - left) * tick / 4
        parts.append(
            f'<text x="{xx:.2f}" y="{bottom + 23}" text-anchor="middle" '
            f'font-size="12" fill="#4b5563">{epoch:.0f}</text>'
        )

    parts.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#374151"/>',
    ])
    for index, (name, metrics) in enumerate(experiments.items()):
        colour = _PALETTE[index]
        coordinates = " ".join(
            f"{x(point.epoch):.2f},{y(point.loss):.2f}" for point in metrics
        )
        parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{colour}" '
            f'stroke-width="2" stroke-linejoin="round"/>'
        )
        legend_x = left + (index % 3) * (right - left) / 3
        legend_y = bottom + 56 + (index // 3) * 22
        parts.extend([
            f'<line x1="{legend_x:.2f}" y1="{legend_y}" '
            f'x2="{legend_x + 18:.2f}" y2="{legend_y}" stroke="{colour}" stroke-width="2"/>',
            f'<text x="{legend_x + 25:.2f}" y="{legend_y + 4}" '
            f'font-size="12" fill="#111827">{escape(name)}</text>',
        ])
    parts.extend([
        f'<text x="{(left + right) / 2:.1f}" y="{bottom + 42}" '
        'text-anchor="middle" font-size="13" fill="#111827">Training epoch</text>',
        f'<text x="22" y="{(top + bottom) / 2:.1f}" '
        f'transform="rotate(-90 22 {(top + bottom) / 2:.1f})" '
        'text-anchor="middle" font-size="13" fill="#111827">Mean squared error</text>',
        '</svg>',
    ])
    return "\n".join(parts) + "\n"


def save_comparison_svg(
    experiments: Mapping[str, Sequence[Metric]], destination: str | Path
) -> Path:
    """Write a comparison plot and return its location."""
    svg = render_comparison_svg(experiments)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return path
