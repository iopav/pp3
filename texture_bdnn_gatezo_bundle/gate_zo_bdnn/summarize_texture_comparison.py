"""Summarize paired fixed/adaptive/gate-ZO texture experiments."""

import argparse
import json
import statistics
from pathlib import Path


METHODS = ("fixed_nozo", "adaptive_nozo", "gatezo")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="*", default=None,
        help="optional seed subset; for example: --seeds 22 23 24 25 26",
    )
    return parser.parse_args()


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    args = parse_args()
    records = {}
    for method in METHODS:
        for path in sorted((args.root / method).glob("seed_*/summary.json")):
            item = json.loads(path.read_text(encoding="utf-8"))
            if args.seeds is not None and int(item["seed"]) not in args.seeds:
                continue
            records.setdefault(int(item["seed"]), {})[method] = item

    print("seed method          exp1_test exp1_h exp2_test exp2_h")
    for seed in sorted(records):
        for method in METHODS:
            item = records[seed].get(method)
            if item is None:
                print("{:4d} {:15s} incomplete".format(seed, method))
                continue
            print("{:4d} {:15s} {:9.4f} {:6d} {:9.4f} {:6d}".format(
                seed, method, item["exp1"]["test"]["accuracy"],
                item["exp1"]["hidden_size"], item["exp2"]["test"]["accuracy"],
                item["exp2"]["hidden_size"],
            ))

    paired = [seed for seed in sorted(records) if all(method in records[seed] for method in METHODS)]
    print("\ncomplete paired seeds:", paired)
    if not paired:
        return
    print("method          exp2_mean exp2_sd hidden_mean hidden_sd")
    aggregates = {}
    for method in METHODS:
        tests = [records[seed][method]["exp2"]["test"]["accuracy"] for seed in paired]
        hidden = [records[seed][method]["exp2"]["hidden_size"] for seed in paired]
        test_mean, test_sd = mean_sd(tests)
        hidden_mean, hidden_sd = mean_sd(hidden)
        aggregates[method] = (tests, hidden)
        print("{:15s} {:.6f} {:.6f} {:11.2f} {:9.2f}".format(
            method, test_mean, test_sd, hidden_mean, hidden_sd,
        ))

    gate_tests, gate_hidden = aggregates["gatezo"]
    print("\nPaired gate-ZO comparisons (exp2):")
    for baseline in ("fixed_nozo", "adaptive_nozo"):
        base_tests, base_hidden = aggregates[baseline]
        deltas = [gate - base for gate, base in zip(gate_tests, base_tests)]
        size_deltas = [gate - base for gate, base in zip(gate_hidden, base_hidden)]
        both = sum(
            gate_acc >= base_acc and gate_h < base_h
            for gate_acc, base_acc, gate_h, base_h
            in zip(gate_tests, base_tests, gate_hidden, base_hidden)
        )
        print(
            "vs {:15s}: mean_acc_delta={:+.6f} mean_hidden_delta={:+.2f} both={}/{}".format(
                baseline, statistics.mean(deltas), statistics.mean(size_deltas), both, len(paired)
            )
        )


if __name__ == "__main__":
    main()
