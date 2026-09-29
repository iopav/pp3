"""Summarize the matched R0 GateZO versus no-GateZO fixed16 experiment."""

import json
import statistics
from pathlib import Path


ROOT = Path("outputs/shd_radlif_r0_gatezo_vs_no_gatezo_5seeds")
SEEDS = (57, 58, 59, 60, 61)
CONFIGS = (
    "R0_fixed_prior_gatezo",
    "R0_fixed_prior_no_gatezo_fixed16",
)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    records = {config: {} for config in CONFIGS}
    print("seed   GateZO accuracy/hidden       no-GateZO accuracy/hidden      delta(GateZO-noZO)")
    for seed in SEEDS:
        for config in CONFIGS:
            path = ROOT / config / f"seed{seed}" / "summary.json"
            if path.is_file():
                row = json.loads(path.read_text())
                if row.get("test"):
                    records[config][seed] = row
        left = records[CONFIGS[0]].get(seed)
        right = records[CONFIGS[1]].get(seed)
        left_text = f"{100*left['test']['accuracy']:.4f}%/h{left['hidden_size']}" if left else "incomplete"
        right_text = f"{100*right['test']['accuracy']:.4f}%/h{right['hidden_size']}" if right else "incomplete"
        delta = (f"{100*(left['test']['accuracy']-right['test']['accuracy']):+.4f}pp"
                 if left and right else "-")
        print(f"{seed:>4}   {left_text:<28} {right_text:<30} {delta}")

    aggregate = []
    print("\nconfig                                  n   mean_test%  sd_test%  mean_hidden")
    for config in CONFIGS:
        rows = list(records[config].values())
        accuracy = [100 * row["test"]["accuracy"] for row in rows]
        hidden = [int(row["hidden_size"]) for row in rows]
        entry = {
            "config": config,
            "n": len(rows),
            "mean_test_percent": statistics.mean(accuracy) if accuracy else None,
            "sd_test_percent": statistics.stdev(accuracy) if len(accuracy) > 1 else None,
            "mean_hidden": statistics.mean(hidden) if hidden else None,
            "selection_mode": (rows[0].get("selection_mode") if rows else None),
        }
        aggregate.append(entry)
        if rows:
            sd = f"{entry['sd_test_percent']:.4f}" if entry["sd_test_percent"] is not None else "NA"
            print(f"{config:<39} {len(rows):>1} {entry['mean_test_percent']:>11.4f} "
                  f"{sd:>8} {entry['mean_hidden']:>12.2f}")
        else:
            print(f"{config:<39} 0 incomplete")

    paired = []
    for seed in SEEDS:
        if seed in records[CONFIGS[0]] and seed in records[CONFIGS[1]]:
            paired.append(100 * (
                records[CONFIGS[0]][seed]["test"]["accuracy"]
                - records[CONFIGS[1]][seed]["test"]["accuracy"]
            ))
    paired_summary = {
        "n": len(paired),
        "mean_gatezo_minus_no_gatezo_pp": statistics.mean(paired) if paired else None,
        "sd_pp": statistics.stdev(paired) if len(paired) > 1 else None,
        "gatezo_wins": sum(value > 0 for value in paired),
        "ties": sum(value == 0 for value in paired),
        "test_used_for_reporting_not_training_or_selection": True,
    }
    if paired:
        sd = f"{paired_summary['sd_pp']:.4f}" if paired_summary["sd_pp"] is not None else "NA"
        print(f"\npaired GateZO-noGateZO={paired_summary['mean_gatezo_minus_no_gatezo_pp']:+.4f} "
              f"+/- {sd} pp; wins={paired_summary['gatezo_wins']}/{len(paired)}")
    output = ROOT / "paired_summary.json"
    output.write_text(json.dumps({
        "seeds": SEEDS,
        "configs": CONFIGS,
        "aggregate": aggregate,
        "paired": paired_summary,
    }, indent=2), encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
