"""Command-line report for paired XOR architecture experiments."""

import argparse
import csv
from pathlib import Path

from synapse.experiments.architecture_comparison import compare_architectures
from synapse.experiments.xor import SAMPLES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=int, default=3)
    parser.add_argument("--candidate", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.8)
    parser.add_argument("--csv", type=Path, help="Optional output CSV path")
    args = parser.parse_args(argv)
    try:
        result = compare_architectures(
            SAMPLES, baseline_hidden_size=args.baseline,
            candidate_hidden_size=args.candidate, seeds=args.seeds,
            epochs=args.epochs, learning_rate=args.learning_rate,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(f"XOR | widths {args.baseline} and {args.candidate} | "
          f"{args.epochs} epochs | learning rate {args.learning_rate}")
    print("Seed       Baseline MSE   Candidate MSE   Difference")
    rows = list(zip(result.seeds, result.baseline.final_losses,
                    result.candidate.final_losses, result.loss_differences))
    for seed, baseline, candidate, delta in rows:
        print(f"{seed:10d} {baseline:14.8f} {candidate:15.8f} {delta:12.8f}")
    print(f"Mean difference: {result.mean_difference:.8f}")
    print(f"Candidate wins / ties / baseline wins: "
          f"{result.candidate_wins}/{result.ties}/{result.baseline_wins}")
    print("Training MSE only; negative differences favour the candidate.")
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("seed", "baseline_width", "candidate_width",
                             "epochs", "learning_rate", "baseline_mse",
                             "candidate_mse", "candidate_minus_baseline"))
            for seed, baseline, candidate, delta in rows:
                writer.writerow((seed, args.baseline, args.candidate,
                                 args.epochs, args.learning_rate,
                                 baseline, candidate, delta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
