"""Render binary classification outcomes as a standalone SVG."""

from pathlib import Path

from synapse.metrics import BinaryMetrics


def render_confusion_svg(metrics: BinaryMetrics) -> str:
    """Show actual classes as rows and predicted classes as columns."""
    counts = (metrics.true_negative, metrics.false_positive,
              metrics.false_negative, metrics.true_positive)
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("Outcome counts must be non-negative integers.")
    total = sum(counts)
    if not total:
        raise ValueError("At least one classification is required.")
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="520" height="340" '
        'viewBox="0 0 520 340" role="img" aria-labelledby="title desc">',
        '<title id="title">Confusion matrix</title>',
        '<desc id="desc">Actual classes are rows; predicted classes are columns. '
        f'True negatives: {counts[0]}; false positives: {counts[1]}; '
        f'false negatives: {counts[2]}; true positives: {counts[3]}.</desc>',
        '<rect width="520" height="340" fill="#ffffff"/>',
        '<g font-family="sans-serif" font-size="14" fill="#222222" text-anchor="middle">',
        '<text x="320" y="25">Predicted class</text>',
        '<text x="230" y="55">Negative</text>',
        '<text x="410" y="55">Positive</text>',
        '<text x="24" y="170" transform="rotate(-90 24 170)">Actual class</text>',
        '<text x="92" y="120">Negative</text>',
        '<text x="92" y="220">Positive</text>',
    ]
    for index, count in enumerate(counts):
        x = 140 + 180 * (index % 2)
        y = 70 + 100 * (index // 2)
        shade = 245 - round(100 * count / max(counts))
        parts.append(f'<rect x="{x}" y="{y}" width="176" height="96" '
                     f'fill="rgb({shade},{shade},{shade})" stroke="#777777"/>')
        parts.append(f'<text x="{x + 88}" y="{y + 42}" font-size="24">{count}</text>')
        parts.append(f'<text x="{x + 88}" y="{y + 68}">{count / total:.1%} of total</text>')
    parts.extend([
        f'<text x="320" y="300">Accuracy: {metrics.accuracy:.1%} | n = {total}</text>',
        '</g></svg>',
    ])
    return "\n".join(parts) + "\n"


def save_confusion_svg(metrics: BinaryMetrics, destination: str | Path) -> Path:
    """Write the matrix to a UTF-8 SVG file."""
    path = Path(destination)
    content = render_confusion_svg(metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
