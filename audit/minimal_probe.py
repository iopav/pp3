"""Offline audit probes; never imports dataset downloaders or original entrypoints.

Run each case in a fresh process because repositories reuse module names.
Only 4 real training examples and 16 sampled timesteps; not benchmark results.
"""
import argparse
import ast
import contextlib
import importlib.util
import io
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"D:\workspace\Avalanche\avalanche-master\dataset")


def load_small(name):
    stem = "20_cla_400steps" if name == "spike" else "texture"
    x = np.load(DATA / name / f"X_train_{stem}.npy", mmap_mode="r")
    y = np.load(DATA / name / f"Y_train_{stem}.npy", mmap_mode="r")
    ids = np.arange(4)
    ts = np.linspace(0, x.shape[1] - 1, 16, dtype=int)
    return torch.tensor(np.array(x[ids][:, ts]), dtype=torch.float32), torch.tensor(np.array(y[ids]), dtype=torch.long), (20 if name == "spike" else 12)


def rsnn(name):
    sys.path.insert(0, str(ROOT / "Integer-Arithmetic-SNN-Learning-main"))
    from models.rec_snn.network import BioGradNetwork
    from core.online_error_functions import LossFunction
    x, y, k = load_small(name)
    out = dict(Vdecay=.5, Vth=1600., Grad_win=1600., Grad_amp=1., Fb_th=1., lr=2**-1, Weight_decay=2**-12)
    hidden = dict(Vdecay=.5, Vth=2000., Grad_win=4000., Grad_amp=1., lr=2**-12, Weight_decay=2**-12, lr_rec=0., Vdecay_rec=.5, fw_factor=1., rec_factor=1.)
    net = BioGradNetwork(x.shape[2], k, [8], dict(out_layer=out, hidden_layer=[hidden]), LossFunction("normal"), torch.device("cpu"), float_mode=False, aligned_weights=False, loss_mode="direct", terminal_stats=False, gradient_clip=256)
    net.norm = 16
    net.writer_batch = False
    for cell in net.hidden_cells + [net.output_cell]:
        cell.writer_batch = cell.writer_spike = False
    net.quantize_network()
    net.prepare_inference_weights()
    before = net.output_cell.forward_func.weight.detach().clone()
    with torch.no_grad():
        result = net.train_online(x.permute(0, 2, 1), torch.nn.functional.one_hot(y, k).float()*128, 5, 128, False, float_mode=False)
        net.train_update_parameter(*result[1:6])
        net.update_inference_weights()
        pred = net.test(x.permute(0, 2, 1))[0]
    return dict(scope="one epoch = one batch of 4 real examples, 16 sampled steps, hidden=8; direct core only", shape=list(x.shape), prediction_shape=list(pred.shape), output_weight_changed=not torch.equal(before, net.output_cell.forward_func.weight), stored_dtype=str(net.output_cell.forward_func.weight.dtype), finite_weights=bool(torch.isfinite(net.output_cell.forward_func.weight).all()), texture_input="unquantized real current, interface probe only" if name == "texture" else "binary spike input")


def bdnn(name):
    sys.path.insert(0, str(ROOT / "BDNNversion1-main"))
    from bdnn1 import bdnn1
    x, y, k = load_small(name)
    model = bdnn1(L_max=1, T_max=4, Lambdas=[1], r=[.99999], nB=3, verbose=1)
    # Admit exactly one of four candidates to isolate candidate-count bookkeeping.
    calls = iter(range(4*k))
    model.inequalityEq = lambda *args: np.array([[1. if next(calls) < k else -1.]])
    w, u, flag = model.sc_Search(x.numpy(), np.eye(k)[y.numpy()])
    returned = int(np.asarray(w).shape[1])
    return dict(scope="one candidate search with exactly one admissible candidate, not training", requested=3, generated=4, admissible=1, returned=returned, flag=flag, defect_reproduced=(flag == 0 and returned < 3))


def ff(name):
    import snntorch as snn
    from snntorch import surrogate
    import torch.nn as nn
    import torch.nn.functional as F
    path = ROOT / "SNN_FF-main/FF_SNN/SNNFF_SHD.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    names = {"LeakyLayer", "superimpose_label"}
    defs = ast.Module(body=[n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names], type_ignores=[])
    ns = dict(torch=torch, nn=nn, F=F, snn=snn, beta=.9, threshold=5., spike_grad4=surrogate.atan(alpha=2), num_steps=16)
    exec(compile(defs, str(path), "exec"), ns)
    x, y, k = load_small(name)
    outcomes = {}
    for key, action in {
        "original_recurrent_layer": lambda: ns["LeakyLayer"](x.shape[2], 500)(x),
        "texture_label_11": lambda: ns["superimpose_label"](torch.ones(1,16,9), torch.tensor([11])),
    }.items():
        try:
            action()
            outcomes[key] = "unexpectedly passed"
        except Exception as exc:
            outcomes[key] = f"{type(exc).__name__}: {exc}"
    return dict(scope="isolated original definitions; no top-level data loader executed", **outcomes)


def gatezo(name):
    sys.path.insert(0, str(ROOT / "texture_bdnn_gatezo_bundle"))
    from gate_zo_bdnn.bdnn1_gatezo import BDNN1GateZO
    x, y, k = load_small(name)
    model = BDNN1GateZO(L_max=1, T_max=4, Lambdas=[1], r=[.99999], nB=1,
        verbose=1, growth_protocol="validation_checkpoint", gate_zo_seed=7,
        gate_zo_verbose=False, compute_device="cpu", compute_batch_size=4)
    targets = np.eye(k)[y.numpy()]
    model.classification(x.numpy(), targets, x.numpy(), targets)
    return dict(scope="one growth attempt, 4 examples, 16 sampled steps; train reused as evaluation solely for smoke, no accuracy claim", summary=model.gatezo_summary if hasattr(model, "gatezo_summary") else str(model.W.shape))


def inventory():
    result = dict(torch=torch.__version__, cuda=torch.cuda.is_available(), modules={m: importlib.util.find_spec(m) is not None for m in ["scipy", "sklearn", "snntorch", "tonic", "spikingjelly", "timm", "tensorboard", "torchvision"]}, arrays=[])
    for name in ("spike", "texture"):
        for p in sorted((DATA/name).glob("*.npy")):
            a = np.load(p, mmap_mode="r")
            item = dict(path=str(p), shape=list(a.shape), dtype=str(a.dtype))
            if a.ndim == 1:
                item["labels"] = np.unique(a).tolist()
            result["arrays"].append(item)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=["inventory", "rsnn", "bdnn", "ff", "gatezo"])
    parser.add_argument("--dataset", choices=["spike", "texture"], default="spike")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    np.random.seed(7)
    start = time.perf_counter()
    log = io.StringIO()
    report = dict(case=args.case, dataset=args.dataset, downloaded=False)
    try:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            report["result"] = inventory() if args.case == "inventory" else globals()[args.case](args.dataset)
        report["status"] = "completed"
    except Exception:
        report.update(status="failed", traceback=traceback.format_exc())
    report.update(seconds=time.perf_counter()-start, log=log.getvalue())
    output = ROOT / "audit/results" / f"{args.case}_{args.dataset}.json"
    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "log"}, ensure_ascii=False))
