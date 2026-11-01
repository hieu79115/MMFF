"""Run independent training seeds and summarize best evaluation accuracy."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--dataset", default="ntu")
    parser.add_argument("--stage", choices=["skeleton", "rgb", "fusion"], default="fusion")
    parser.add_argument("--fusion_type", choices=["cmaf", "sum", "average", "concat"], default="cmaf")
    parser.add_argument("--cross_attention", choices=["normal", "none", "reversed"], default="normal")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry_run", action="store_true")
    args, train_args = parser.parse_known_args()

    metrics = []
    for seed in args.seeds:
        command = [
            args.python,
            "train.py",
            "--seed",
            str(seed),
            "--split_seed",
            str(args.split_seed),
            "--dataset",
            args.dataset,
            "--stage",
            args.stage,
            "--fusion_type",
            args.fusion_type,
            "--cross_attention",
            args.cross_attention,
            *train_args,
        ]
        print("$", " ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)
            metrics_path = Path(
                f"metrics_{args.stage}_{args.dataset}_{args.fusion_type}_attn-{args.cross_attention}_seed{seed}.json"
            )
            if not metrics_path.exists():
                raise FileNotFoundError(f"No metrics file found at {metrics_path}")
            with metrics_path.open(encoding="utf-8") as metrics_file:
                metrics.append(json.load(metrics_file))

    if args.dry_run:
        return
    accuracies = [item["best_accuracy_percent"] for item in metrics]
    summary = {
        "seeds": args.seeds,
        "split_seed": args.split_seed,
        "accuracies_percent": accuracies,
        "mean_percent": statistics.mean(accuracies),
        "std_percent": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
    }
    summary_path = Path("multiseed_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Accuracy: {summary['mean_percent']:.2f} ± "
        f"{summary['std_percent']:.2f}% (seeds: {', '.join(map(str, args.seeds))})"
    )
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
