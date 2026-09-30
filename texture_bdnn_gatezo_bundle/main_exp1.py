#gpu or not
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from bdnn1 import bdnn1
from inputsdata import load_data
from seed_utils import set_global_seed
import time

from config import SCN_DEFAULTS
MODEL_CONFIG = dict(SCN_DEFAULTS)
CONFIG_SEED = MODEL_CONFIG.pop('seed', None)
SEED = set_global_seed(CONFIG_SEED)
print("Using seed:", SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Producing data...")

X_train_path, Y_train_path, X_test_path, Y_test_path = (
    './data/old6classes/X_train_6cla_old.npy',
    './data/old6classes/Y_train_6cla_old.npy',
    './data/old6classes/X_test_6cla_old.npy',
    './data/old6classes/Y_test_6cla_old.npy',

)
print("Loading data...")
original_classes,classes = 0, 6


X_train, Y_train, X_test, Y_test = (
    load_data(X_train_path, Y_train_path, X_test_path, Y_test_path, original_classes, classes))
print('Y_test is', Y_test)
class_names = np.array(['1', '2', '3','4','5','6'])

start = time.time()
M = bdnn1(**MODEL_CONFIG)
ErrorList, RateList, RateList2, timelist, suitW, suitU, Beta = M.classification(X_train, Y_train, X_test, Y_test)
end = time.time()
np.save('suitW_old_6cla.npy', suitW)
np.save('suitU_old_6cla.npy', suitU)
np.save('suitB_old_6cla.npy', Beta)
np.save('Ratelist_for_6_old_classes.npy', RateList)
np.save('RateList2_for_6_old_classes.npy', RateList2)

print('the size of weight is : '+ str(suitW.size()))
print("Time taken = ", (end - start) / 3600)
#保存模型

np.save('error_old_6cla.npy', ErrorList)
np.save('rate_old_6cla.npy', RateList)
import numpy as np
import matplotlib.pyplot as plt

# Load arrays
RateList = np.load("Ratelist_for_6_old_classes.npy").flatten()
RateList2 = np.load("RateList2_for_6_old_classes.npy").flatten()

# Plot training accuracy
plt.figure(figsize=(8, 5))
plt.plot(RateList, marker='o', label="Training Accuracy")
plt.xlabel("No. of neurons")
plt.ylabel("Accuracy")

plt.ylim(0.9, 1.01)  # zoom in near the accuracy range
plt.grid(True)
plt.legend()
plt.savefig('accuracy_old_6cla.png')
if os.environ.get("BDNN_NO_SHOW", "0") != "1":
    plt.show()
plt.close()

# Plot test accuracy
plt.figure(figsize=(8, 5))
plt.plot(RateList2, marker='s', color="orange", label="Test Accuracy")
plt.xlabel("No. of neurons")
plt.ylabel("Accuracy")


plt.ylim(0.9, 1.01)  # zoom in near the accuracy range
plt.grid(True)
plt.legend()
plt.savefig('accuracy2_old_6cla.png')
if os.environ.get("BDNN_NO_SHOW", "0") != "1":
    plt.show()
plt.close()
