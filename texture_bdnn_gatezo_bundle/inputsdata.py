import numpy as np
import torch

def load_data(X_train_path, Y_train_path, X_test_path, Y_test_path,
              original_classes=5, classes=5):

    # Load .npy files
    X_train = np.load(X_train_path)
    Y_train = np.load(Y_train_path)
    X_test = np.load(X_test_path)
    Y_test = np.load(Y_test_path)

    # Convert to torch tensors
    X_train = torch.from_numpy(X_train).float()
    Y_train = torch.from_numpy(Y_train).long() - original_classes
    X_test = torch.from_numpy(X_test).float()
    Y_test = torch.from_numpy(Y_test).long() - original_classes

    # One-hot encoding
    Y_train = torch.zeros(Y_train.size(0), classes).scatter_(1, Y_train.unsqueeze(1), 1)
    Y_test = torch.zeros(Y_test.size(0), classes).scatter_(1, Y_test.unsqueeze(1), 1)

    return X_train, Y_train, X_test, Y_test
