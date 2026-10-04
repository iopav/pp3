"""Run stage 2 from the gate-ZO stage-1 checkpoint."""

import json
import os
from pathlib import Path

import numpy as np

from inputsdata import load_data
from seed_utils import set_global_seed

from .bdnn2_gatezo import BDNN2GateZO
from .config_gatezo import GATE_ZO_DEFAULTS


ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = os.environ.get("BDNN_GATE_ZO_RUN", "current")
BASE = ROOT / "outputs" / "gate_zo_bdnn" / RUN_NAME
OUTPUT = BASE / "exp2"


def main():
    config = dict(GATE_ZO_DEFAULTS)
    default_seed = config.pop("seed", None)
    seed = set_global_seed(default_seed)
    config["gate_zo_seed"] = seed
    X, T, X_test, T_test = load_data(
        ROOT / "data/new6classes/X_train_6cla_new.npy",
        ROOT / "data/new6classes/Y_train_6cla_new.npy",
        ROOT / "data/new6classes/X_test_6cla_new.npy",
        ROOT / "data/new6classes/Y_test_6cla_new.npy", 6, 6,
    )
    model = BDNN2GateZO(**config)
    result = model.classification(
        X, T, X_test, T_test,
        BASE / "exp1/suitW_old_6cla_gatezo.npy",
        BASE / "exp1/suitU_old_6cla_gatezo.npy",
    )
    _, train_rates, test_rates, times, W, U, beta = result
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT / "suitW_new_6cla_gatezo.npy", W)
    np.save(OUTPUT / "suitU_new_6cla_gatezo.npy", U)
    np.save(OUTPUT / "suitB_new_6cla_gatezo.npy", beta)
    np.save(OUTPUT / "train_rates.npy", train_rates)
    np.save(OUTPUT / "test_rates.npy", test_rates)
    np.save(OUTPUT / "time_hours.npy", times)
    summary = {"seed": seed, **model.gate_zo_summary, "history": model.gate_zo_history}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(model.gate_zo_summary, indent=2))


if __name__ == "__main__":
    main()
