"""Run paired stage-1/stage-2 gate-ZO experiments for multiple seeds."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "gate_zo_bdnn"


def seeds_from_environment():
    raw = os.environ.get("BDNN_SEEDS", "22")
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def run(module, seed, run_name, log_path):
    environment = os.environ.copy()
    environment.update({
        "BDNN_SEED": str(seed),
        "BDNN_GATE_ZO_RUN": run_name,
        "BDNN_NO_SHOW": "1",
    })
    completed = subprocess.run(
        [sys.executable, "-m", module], cwd=ROOT, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        print(completed.stdout)
        raise SystemExit(completed.returncode)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds_from_environment():
        run_name = "seed_{}".format(seed)
        run_dir = OUTPUT_ROOT / run_name
        print("=== gate-ZO seed {}: stage 1 ===".format(seed))
        run(
            "gate_zo_bdnn.main_exp1_gatezo", seed, run_name,
            run_dir / "exp1/exp1.log",
        )
        print("=== gate-ZO seed {}: stage 2 ===".format(seed))
        run(
            "gate_zo_bdnn.main_exp2_gatezo", seed, run_name,
            run_dir / "exp2/exp2.log",
        )
        stage1 = json.loads((run_dir / "exp1/summary.json").read_text(encoding="utf-8"))
        stage2 = json.loads((run_dir / "exp2/summary.json").read_text(encoding="utf-8"))
        row = {
            "seed": seed,
            "exp1_test_accuracy": stage1["test"]["accuracy"],
            "exp1_hidden": stage1["hidden_size"],
            "exp1_accepted_rounds": stage1["accepted_rounds"],
            "exp2_test_accuracy": stage2["test"]["accuracy"],
            "exp2_hidden": stage2["hidden_size"],
            "exp2_accepted_rounds": stage2["accepted_rounds"],
            "test_used_for_selection": False,
        }
        rows.append(row)
        print(row)
    destination = OUTPUT_ROOT / "seed_sweep_summary.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved summary:", destination)


if __name__ == "__main__":
    main()
