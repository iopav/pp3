#gpu or not
import torch
import numpy as np
import matplotlib.pyplot as plt
from bdnn1 import bdnn1
from inputsdata import load_data
import time

from config import SCN_DEFAULTS

# GPU board power monitor: reports measured W and estimated J in the training window.
# Requires pynvml; this measures whole-GPU power, not CPU training energy.
import threading

class GPUBoardEnergy:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None
        self.nv = None
        self.error = None

    def _sample(self):
        try:
            power_w = self.nv.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            self.samples.append((time.perf_counter(), power_w))
        except Exception as exc:
            self.error = str(exc)
            self.stop_event.set()

    def _poll(self):
        while not self.stop_event.wait(self.interval):
            self._sample()

    def __enter__(self):
        self.start = time.perf_counter()
        try:
            import pynvml
            self.nv = pynvml
            self.nv.nvmlInit()
            # Match the GPU visible to this PyTorch process, including Slurm mappings.
            if not torch.cuda.is_available():
                raise RuntimeError('No CUDA GPU visible')
            uuid = str(torch.cuda.get_device_properties(0).uuid)
            if not uuid.startswith(('GPU-', 'MIG-')):
                uuid = 'GPU-' + uuid
            self.handle = self.nv.nvmlDeviceGetHandleByUUID(uuid)
            self._sample()
            self.thread = threading.Thread(target=self._poll, daemon=True)
            self.thread.start()
        except Exception as exc:
            self.error = str(exc)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self._sample() if self.nv is not None and hasattr(self, 'handle') and self.error is None else None
        self.end = time.perf_counter()
        if self.nv is not None:
            self.nv.nvmlShutdown()

    def report(self):
        duration_s = self.end - self.start
        if self.error is not None or len(self.samples) < 2:
            print(f'GPU energy unavailable: {self.error or "Insufficient samples"}')
            return
        energy_j = sum(
            (t2 - t1) * (p1 + p2) / 2
            for (t1, p1), (t2, p2) in zip(self.samples, self.samples[1:])
        )
        measured_s = self.samples[-1][0] - self.samples[0][0]
        average_w = energy_j / measured_s if measured_s > 0 else float('nan')
        print(f'GPU monitored time: {measured_s:.3f} s')
        print(f'GPU average board power: {average_w:.2f} W')
        print(f'GPU estimated board energy: {energy_j:.2f} J')
        print('Note: whole-GPU board power; includes idle draw and other GPU activity.')

import os
runtime = None
if os.environ.get('PP3_CONFIG'):
    from pp3.native_runtime import from_environment
    runtime = from_environment()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Producing data...")

data_paths =r"pp3/data/texture-mini/texture_spikesT450_T77.npz"

print("Loading data...")
with np.load(data_paths) as data:
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_test = data["X_test"]
    Y_test = data["Y_test"]
X_train = torch.from_numpy(X_train).float()
Y_train = torch.from_numpy(Y_train).long()
X_test = torch.from_numpy(X_test).float()
Y_test = torch.from_numpy(Y_test).long()
original_classes,classes = 0, 12


if runtime:
    X_train, labels_train, X_test, labels_test = runtime.arrays()
    classes = runtime.config['data']['classes']
    Y_train = torch.nn.functional.one_hot(labels_train, classes).float()
    Y_test = torch.nn.functional.one_hot(labels_test, classes).float()
    settings = runtime.config['model']
    SCN_DEFAULTS = dict(SCN_DEFAULTS, L_max=settings['epochs'], T_max=settings['pool_size'],
                       nB=settings['growth_step'], tol=settings['tol'],
                       Lambdas=settings['lambdas'], r=settings['r'])
    # This original BDNN implementation computes on CPU; do not report GPU model energy.
    runtime.device = torch.device('cpu')
    os.chdir(runtime.output)
else:
    X_train_path, Y_train_path, X_test_path, Y_test_path = data_paths
    X_train, Y_train, X_test, Y_test = load_data(
        X_train_path, Y_train_path, X_test_path, Y_test_path, original_classes, classes)
print('Y_test is', Y_test)
class_names = np.array(['1', '2', '3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])[:classes]

# Power monitoring starts before model construction and stops after classification.
with GPUBoardEnergy(interval=0.1) as gpu_meter:
    start = time.perf_counter()
    M = bdnn1(**SCN_DEFAULTS)
    if runtime:
        M.alpha, M.beta, M._pp3_runtime = settings['alpha'], settings['beta'], runtime
    ErrorList, RateList, RateList2, timelist, suitW, suitU, Beta = M.classification(X_train, Y_train, X_test, Y_test)
    end = time.perf_counter()
gpu_meter.report()
if runtime:
    runtime.finish(dict(W=suitW, U=suitU, Beta=Beta))
# np.save('suitW_old_20cla.npy', suitW)
# np.save('suitU_old_20cla.npy', suitU)
# np.save('suitB_old_20cla.npy', Beta)
# np.save('Ratelist_for_20_old_classes.npy', RateList)
# np.save('RateList2_for_20_old_classes.npy', RateList2)

# print('the shape of weight is : '+ str(suitW.shape))
print("Time taken = ", (end - start) / 3600)
#保存模型

# np.save('error_old_6cla.npy', ErrorList)
# np.save('rate_old_6cla.npy', RateList)
# import numpy as np
# import matplotlib.pyplot as plt

# Load arrays
# RateList = np.load("Ratelist_for_6_old_classes.npy").flatten()
# RateList2 = np.load("RateList2_for_6_old_classes.npy").flatten()

# Plot training accuracy
# plt.figure(figsize=(8, 5))
# plt.plot(RateList, marker='o', label="Training Accuracy")
# plt.xlabel("No. of neurons")
# plt.ylabel("Accuracy")

# plt.ylim(0.9, 1.01)  # zoom in near the accuracy range
# plt.grid(True)
# plt.legend()
# plt.show()

# # Plot test accuracy
# plt.figure(figsize=(8, 5))
# plt.plot(RateList2, marker='s', color="orange", label="Test Accuracy")
# plt.xlabel("No. of neurons")
# plt.ylabel("Accuracy")


# plt.ylim(0.9, 1.01)  # zoom in near the accuracy range
# plt.grid(True)
# plt.legend()
# plt.show()

# #save plt
# plt.savefig('accuracy_old_6cla.png')
# plt.savefig('accuracy2_old_6cla.png')
# plt.close()
