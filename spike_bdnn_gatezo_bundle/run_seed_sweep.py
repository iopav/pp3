import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)


def parse_seeds():
    raw = os.environ.get("BDNN_SEEDS", "2024,2025,2026")
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def run_script(script, seed):
    env = os.environ.copy()
    env["BDNN_SEED"] = str(seed)
    env["BDNN_NO_SHOW"] = "1"
    start = time.time()
    completed = subprocess.run(
        [str(PYTHON), script],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        check=False,
    )
    return completed.returncode, completed.stdout, time.time() - start


def exp1_max_shape():
    raw = os.environ.get("BDNN_EXP1_MAX_W_SHAPE", "9,12")
    left, right = [int(item.strip()) for item in raw.split(",", 1)]
    return left, right


def should_skip_exp2(old_w_shape):
    max_left, max_right = exp1_max_shape()
    return old_w_shape[0] > max_left or old_w_shape[1] > max_right


def array_summary(path):
    arr = np.load(str(ROOT / path), allow_pickle=True)
    return arr.shape, float(np.nanmax(arr)) if arr.size else float("nan")


EXP1_ARTIFACTS = [
    "suitW_old_6cla.npy",
    "suitU_old_6cla.npy",
    "suitB_old_6cla.npy",
    "Ratelist_for_6_old_classes.npy",
    "RateList2_for_6_old_classes.npy",
    "error_old_6cla.npy",
    "rate_old_6cla.npy",
    "accuracy_old_6cla.png",
    "accuracy2_old_6cla.png",
]

EXP2_ARTIFACTS = [
    "suitW_for_6_new_classes.npy",
    "suitU_for_6_new_classes.npy",
    "suitBeta_for_6_new_classes.npy",
    "Ratelist_for_6_new_classes.npy",
    "RateList2_for_6_new_classes.npy",
    "accuracy_new_6cla.png",
    "accuracy2_new_6cla.png",
]


def copy_artifacts(names, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = ROOT / name
        if src.exists():
            shutil.copy2(str(src), str(destination / name))


def collect_result(seed, exp1_seconds, exp2_seconds=None, exp2_skipped=False, skip_reason=""):
    old_rate_shape, old_rate_max = array_summary("RateList2_for_6_old_classes.npy")
    old_w = np.load(str(ROOT / "suitW_old_6cla.npy"), allow_pickle=True)
    old_u = np.load(str(ROOT / "suitU_old_6cla.npy"), allow_pickle=True)
    row = {
        "seed": seed,
        "exp1_test_acc": old_rate_max,
        "exp1_hidden": old_w.shape[1],
        "exp1_w_shape": str(old_w.shape),
        "exp1_u_shape": str(old_u.shape),
        "exp1_rate_len": old_rate_shape[1] if len(old_rate_shape) > 1 else old_rate_shape[0],
        "exp1_seconds": round(exp1_seconds, 3),
        "exp2_skipped": exp2_skipped,
        "skip_reason": skip_reason,
    }
    if exp2_skipped:
        row.update({
            "exp2_test_acc": "",
            "exp2_hidden": "",
            "exp2_w_shape": "",
            "exp2_u_shape": "",
            "exp2_rate_len": "",
            "exp2_seconds": "",
        })
    else:
        new_rate_shape, new_rate_max = array_summary("RateList2_for_6_new_classes.npy")
        new_w = np.load(str(ROOT / "suitW_for_6_new_classes.npy"), allow_pickle=True)
        new_u = np.load(str(ROOT / "suitU_for_6_new_classes.npy"), allow_pickle=True)
        row.update({
            "exp2_test_acc": new_rate_max,
            "exp2_hidden": new_w.shape[1],
            "exp2_w_shape": str(new_w.shape),
            "exp2_u_shape": str(new_u.shape),
            "exp2_rate_len": new_rate_shape[1] if len(new_rate_shape) > 1 else new_rate_shape[0],
            "exp2_seconds": round(exp2_seconds, 3),
        })
    return row


def main():
    seeds = parse_seeds()
    output_dir = ROOT / "outputs" / "seed_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        seed_dir = output_dir / "seed_{}".format(seed)
        exp1_dir = seed_dir / "exp1"
        exp2_dir = seed_dir / "exp2"
        seed_dir.mkdir(parents=True, exist_ok=True)

        print("=== seed {} exp1 ===".format(seed))
        code1, out1, seconds1 = run_script("main_exp1.py", seed)
        (exp1_dir / "exp1.log").parent.mkdir(parents=True, exist_ok=True)
        (exp1_dir / "exp1.log").write_text(out1, encoding="utf-8", errors="replace")
        if code1 != 0:
            print(out1)
            raise SystemExit(code1)
        copy_artifacts(EXP1_ARTIFACTS, exp1_dir)

        old_w = np.load(str(ROOT / "suitW_old_6cla.npy"), allow_pickle=True)
        if should_skip_exp2(old_w.shape):
            max_left, max_right = exp1_max_shape()
            reason = "exp1 W shape {} exceeds max ({}, {})".format(old_w.shape, max_left, max_right)
            (seed_dir / "SKIPPED_EXP2.txt").write_text(reason + "\n", encoding="utf-8")
            row = collect_result(seed, seconds1, exp2_skipped=True, skip_reason=reason)
            rows.append(row)
            print(row)
            continue

        print("=== seed {} exp2 ===".format(seed))
        code2, out2, seconds2 = run_script("main_exp2.py", seed)
        (exp2_dir / "exp2.log").parent.mkdir(parents=True, exist_ok=True)
        (exp2_dir / "exp2.log").write_text(out2, encoding="utf-8", errors="replace")
        if code2 != 0:
            print(out2)
            raise SystemExit(code2)
        copy_artifacts(EXP2_ARTIFACTS, exp2_dir)

        row = collect_result(seed, seconds1, seconds2)
        rows.append(row)
        print(row)

    summary_path = output_dir / "seed_sweep_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
