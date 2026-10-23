"""Minimal environment/GPU check. No dataset access or training outputs."""
import os
import socket
import sys


def main():
    print("Host:", socket.gethostname(), flush=True)
    print("Python:", sys.executable, flush=True)
    for name in ("CONDA_DEFAULT_ENV", "CONDA_PREFIX", "CUDA_VISIBLE_DEVICES"):
        print(f"{name}: {os.environ.get(name, '<unset>')}", flush=True)

    print("[1/3] Import NumPy", flush=True)
    import numpy as np
    print("NumPy:", np.__version__, np.__file__, flush=True)

    print("[2/3] Import PyTorch and convert NumPy array", flush=True)
    import torch
    print("PyTorch:", torch.__version__, torch.__file__, flush=True)
    print("PyTorch CUDA:", torch.version.cuda, flush=True)
    x = torch.from_numpy(np.ones((2, 2), dtype=np.float32))

    print("[3/3] CUDA forward/backward", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run this check inside a GPU allocation.")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    x = x.to("cuda:0").requires_grad_()
    x.square().sum().backward()
    torch.cuda.synchronize()
    if not torch.equal(x.grad, torch.full_like(x, 2.0)):
        raise RuntimeError("Unexpected gradient in CUDA check.")
    print("PASS: NumPy, PyTorch, NumPy bridge, CUDA forward/backward", flush=True)


if __name__ == "__main__":
    main()
