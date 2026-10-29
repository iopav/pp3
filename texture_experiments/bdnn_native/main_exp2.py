#gpu or not
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Producing data...")
import numpy as np
from bdnn2 import bdnn2
from inputsdata import load_data


X_train_path, Y_train_path, X_test_path, Y_test_path = (
    './data/new6classes/X_train_6cla_new.npy',
    './data/new6classes/Y_train_6cla_new.npy',
    './data/new6classes/X_test_6cla_new.npy',
    './data/new6classes/Y_test_6cla_new.npy',
)
original_classes,classes = 6, 6
X_train, Y_train, X_test, Y_test = (
    load_data(X_train_path, Y_train_path, X_test_path, Y_test_path, original_classes, classes))


W_path, U_path, Beta_path = 'suitW_old_6cla.npy', 'suitU_old_6cla.npy', 'suitB_old_6cla.npy'

from config import SCN_DEFAULTS

M = bdnn2(**SCN_DEFAULTS)

class_names = np.array(['1', '2', '3','4','5','6'])


ErrorList, RateList, RateList2, timeList, suitW, suitU, suitBeta =\
    M.classification(X_train, Y_train,  X_test, Y_test, W_path, U_path )


np.save('suitW_for_6_new_classes.npy', suitW)
np.save('suitU_for_6_new_classes.npy', suitU)
np.save('suitBeta_for_6_new_classes.npy', suitBeta)
np.save('Ratelist_for_6_new_classes.npy', RateList)
np.save('RateList2_for_6_new_classes.npy', RateList2)
import numpy as np
import matplotlib.pyplot as plt

# Load arrays
RateList = np.load("Ratelist_for_6_new_classes.npy").flatten()
RateList2 = np.load("RateList2_for_6_new_classes.npy").flatten()

# Plot training accuracy
plt.figure(figsize=(8, 5))
plt.plot(RateList, marker='o', label="Training Accuracy")
plt.xlabel("No. of neurons")
plt.ylabel("Accuracy")

plt.ylim(0.9, 1.01)  # zoom in near the accuracy range
plt.grid(True)
plt.legend()
plt.show()

# Plot test accuracy
plt.figure(figsize=(8, 5))
plt.plot(RateList2, marker='s', color="orange", label="Test Accuracy")
plt.xlabel("No. of neurons")
plt.ylabel("Accuracy")

plt.ylim(0.9, 1.01)  # zoom in near the accuracy range
plt.grid(True)
plt.legend()
plt.show()

#save plt
plt.savefig('accuracy_new_6cla.png')
plt.savefig('accuracy2_new_6cla.png')



