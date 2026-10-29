#gpu or not
import torch
import numpy as np
import matplotlib.pyplot as plt
from bdnn1 import bdnn1
from inputsdata import load_data
import time

from config import SCN_DEFAULTS
import os
runtime = None
if os.environ.get('PP3_CONFIG'):
    from pp3.native_runtime import from_environment
    runtime = from_environment()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Producing data...")

data_paths = (
    # './data/spike/X_train_20cla_400steps.npy',s
    # './data/spike/Y_train_20cla_400steps.npy',
    # './data/spike/X_test_20cla_400steps.npy',
    # './data/spike/Y_test_20cla_400steps.npy',

)
print("Loading data...")
original_classes,classes = 0, 20


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
elif os.environ.get('TEXTURE_BDNN_CONFIG'):
    # Dataset adaptation only. SCN_DEFAULTS and classification stay native.
    from texture_input import load_texture_data
    X_train, Y_train, X_test, Y_test = load_texture_data()
    classes = Y_train.shape[1]
else:
    X_train_path, Y_train_path, X_test_path, Y_test_path = data_paths
    X_train, Y_train, X_test, Y_test = load_data(
        X_train_path, Y_train_path, X_test_path, Y_test_path, original_classes, classes)
print('Y_test is', Y_test)
class_names = np.array(['1', '2', '3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])

start = time.time()
M = bdnn1(**SCN_DEFAULTS)
texture_metrics = None
if os.environ.get('TEXTURE_BDNN_CONFIG'):
    from texture_input import make_metrics
    texture_metrics = make_metrics(M)
    M._pp3_runtime = texture_metrics
if runtime:
    M.alpha, M.beta, M._pp3_runtime = settings['alpha'], settings['beta'], runtime
ErrorList, RateList, RateList2, timelist, suitW, suitU, Beta = M.classification(X_train, Y_train, X_test, Y_test)
end = time.time()
if texture_metrics is not None:
    texture_metrics.finish(dict(W=suitW, U=suitU, Beta=Beta))
if runtime:
    runtime.finish(dict(W=suitW, U=suitU, Beta=Beta))
np.save('suitW_old_20cla.npy', suitW)
np.save('suitU_old_20cla.npy', suitU)
np.save('suitB_old_20cla.npy', Beta)
np.save('Ratelist_for_20_old_classes.npy', RateList)
np.save('RateList2_for_20_old_classes.npy', RateList2)

print('the shape of weight is : '+ str(suitW.shape))
print("Time taken = ", (end - start) / 3600)
#保存模型

np.save('error_old_6cla.npy', ErrorList)
np.save('rate_old_6cla.npy', RateList)
import numpy as np
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
