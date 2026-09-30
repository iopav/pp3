"""Summarize constrained gate-ZO using validation loss for model selection."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


CONFIGS = ("eps0000", "eps0001", "eps0003", "eps0005", "eps0010")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("outputs/texture_gatezo_constrained_search"),
    )
    parser.add_argument(
        "--baseline-root", type=Path,
        default=Path(
            "outputs/texture_gatezo_preselect5/validation_checkpoint/"
            "adaptive_nozo"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(22, 27)))
    parser.add_argument("--loss-margin", type=float, default=0.002)
    return parser.parse_args()


def read(path):
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))["exp2"]
    validation = data["structure_validation"]
    return {
        "val_loss": float(validation["loss"]),
        "val_acc": float(validation["accuracy"]),
        "test": float(data["test"]["accuracy"]),
        "hidden": int(data["hidden_size"]),
    }


def aggregate(rows):
    result = {}
    for key in ("val_loss", "val_acc", "test", "hidden"):
        items = [row[key] for row in rows]
        result["mean_" + key] = mean(items)
        result["sd_" + key] = stdev(items) if len(items) > 1 else 0.0
    return result


def main():
    args = arguments()
    rows = {"adaptive_nozo": {}}
    for seed in args.seeds:
        item = read(args.baseline_root / f"seed_{seed}" / "summary.json")
        if item is not None:
            rows["adaptive_nozo"][seed] = item

    for config in CONFIGS:
        rows[config] = {}
        for seed in args.seeds:
            path = (
                args.root / config / "validation_checkpoint" / "gatezo" /
                f"seed_{seed}" / "summary.json"
            )
            item = read(path)
            if item is not None:
                rows[config][seed] = item

    common = sorted(set(args.seeds).intersection(*(set(item) for item in rows.values())))
    print("common paired seeds:", common)
    print("completed tasks: {}/{}".format(
        sum(len(rows[name]) for name in CONFIGS), len(CONFIGS) * len(args.seeds)
    ))
    if not common:
        print("No complete paired seed set yet.")
        return

    stats = {
        name: aggregate([values[seed] for seed in common])
        for name, values in rows.items()
    }
    base = stats["adaptive_nozo"]
    eligible = [
        name for name in CONFIGS
        if stats[name]["mean_val_loss"] <= base["mean_val_loss"] + args.loss_margin
    ]
    recommended = min(
        eligible,
        key=lambda name: (stats[name]["mean_hidden"], stats[name]["mean_val_loss"]),
    ) if eligible else min(CONFIGS, key=lambda name: stats[name]["mean_val_loss"])

    print(
        "config        val_loss  d_loss val_acc test_acc  d_test mean_h    d_h "
        "loss_ok recommended"
    )
    for name in ("adaptive_nozo",) + CONFIGS:
        item = stats[name]
        print(
            f"{name:13s} {item['mean_val_loss']:.6f} "
            f"{item['mean_val_loss'] - base['mean_val_loss']:+.5f} "
            f"{item['mean_val_acc']:.4f} {item['mean_test']:.6f} "
            f"{item['mean_test'] - base['mean_test']:+.4f} "
            f"{item['mean_hidden']:6.2f} "
            f"{item['mean_hidden'] - base['mean_hidden']:+6.2f} "
            f"{str(name == 'adaptive_nozo' or name in eligible):>7s} "
            f"{str(name == recommended):>11s}"
        )

    print()
    print(
        "selection_rule=validation loss no more than {:.4f} above adaptive, "
        "then minimum hidden".format(args.loss_margin)
    )
    print("recommended_config={}".format(recommended))
    print("IMPORTANT: test accuracy is audit-only and is not used to select epsilon.")


if __name__ == "__main__":
    main()
