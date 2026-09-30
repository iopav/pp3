"""Summarize the one-estimate gate-ZO trade-off search without test tuning."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


CONFIGS = (
    "k2_l0000",
    "k2_l00005",
    "k2_l0001",
    "k2_l0002",
    "k2_l0003",
    "k3_l00005",
    "k3_l0001",
    "k3_l0002",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("outputs/texture_gatezo_tradeoff_search"),
    )
    parser.add_argument(
        "--baseline-root", type=Path,
        default=Path(
            "outputs/texture_gatezo_preselect5/validation_checkpoint/"
            "adaptive_nozo"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(22, 27)))
    parser.add_argument(
        "--accuracy-margin", type=float, default=0.002,
        help="allowed mean validation-accuracy deficit versus adaptive_nozo",
    )
    return parser.parse_args()


def load_summary(path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(summary):
    exp2 = summary["exp2"]
    validation = exp2.get("structure_validation") or {}
    return {
        "val": float(validation.get("accuracy", float("nan"))),
        "test": float(exp2["test"]["accuracy"]),
        "hidden": int(exp2["hidden_size"]),
    }


def aggregate(rows):
    return {
        "mean_val": mean(row["val"] for row in rows),
        "sd_val": stdev(row["val"] for row in rows) if len(rows) > 1 else 0.0,
        "mean_test": mean(row["test"] for row in rows),
        "sd_test": stdev(row["test"] for row in rows) if len(rows) > 1 else 0.0,
        "mean_hidden": mean(row["hidden"] for row in rows),
        "sd_hidden": stdev(row["hidden"] for row in rows) if len(rows) > 1 else 0.0,
    }


def main():
    args = parse_args()
    methods = {}
    baseline = {}
    for seed in args.seeds:
        item = load_summary(args.baseline_root / f"seed_{seed}" / "summary.json")
        if item is not None:
            baseline[seed] = metrics(item)
    methods["adaptive_nozo"] = baseline

    for config in CONFIGS:
        rows = {}
        for seed in args.seeds:
            path = (
                args.root / config / "validation_checkpoint" /
                "gatezo" / f"seed_{seed}" / "summary.json"
            )
            item = load_summary(path)
            if item is not None:
                rows[seed] = metrics(item)
        methods[config] = rows

    common = sorted(set(args.seeds).intersection(*(set(rows) for rows in methods.values())))
    print("common paired seeds:", common)
    print("completed tasks: {}/{}".format(
        sum(len(methods[name]) for name in CONFIGS), len(CONFIGS) * len(args.seeds)
    ))
    if not common:
        print("No complete paired seed set yet.")
        return

    stats = {
        name: aggregate([rows[seed] for seed in common])
        for name, rows in methods.items()
    }
    baseline_stats = stats["adaptive_nozo"]
    eligible = {
        name for name in CONFIGS
        if stats[name]["mean_val"] >= baseline_stats["mean_val"] - args.accuracy_margin
    }
    recommended = min(
        eligible,
        key=lambda name: (stats[name]["mean_hidden"], -stats[name]["mean_val"]),
    ) if eligible else max(CONFIGS, key=lambda name: stats[name]["mean_val"])

    print(
        "config          mean_val  d_val   mean_test  d_test  mean_h   d_h "
        " val_ok recommended"
    )
    for name in ("adaptive_nozo",) + CONFIGS:
        row = stats[name]
        d_val = row["mean_val"] - baseline_stats["mean_val"]
        d_test = row["mean_test"] - baseline_stats["mean_test"]
        d_hidden = row["mean_hidden"] - baseline_stats["mean_hidden"]
        print(
            f"{name:15s} {row['mean_val']:.6f} {d_val:+.4f} "
            f"{row['mean_test']:.6f} {d_test:+.4f} "
            f"{row['mean_hidden']:7.2f} {d_hidden:+6.2f} "
            f"{str(name in eligible):>6s} {str(name == recommended):>11s}"
        )

    print()
    print("selection_rule=validation accuracy within {:.4f} of adaptive, then minimum hidden".format(
        args.accuracy_margin
    ))
    print("recommended_config={}".format(recommended))
    print("IMPORTANT: test accuracy is displayed for audit only, not used for selection.")


if __name__ == "__main__":
    main()
