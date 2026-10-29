import numpy as np


def split_test_set_by_class(X_test, Y_test, val_ratio=0.5, shuffle=True, random_seed=None):
    """
    按类别将测试集划分为验证集和新测试集，每类各选 val_ratio 比例样本用于验证集。

    参数：
        X_test (np.ndarray): 测试集样本数据
        Y_test (np.ndarray): 测试集标签
        val_ratio (float): 每类样本分配给验证集的比例，默认为 0.5
        shuffle (bool): 是否打乱验证集和新测试集
        random_seed (int or None): 随机种子，确保可复现

    返回：
        X_val (np.ndarray): 验证集样本
        Y_val (np.ndarray): 验证集标签
        X_test_new (np.ndarray): 新测试集样本
        Y_test_new (np.ndarray): 新测试集标签
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    Y_test = Y_test.astype(int)
    classes = np.unique(Y_test)

    X_val_list, Y_val_list = [], []
    X_test_new_list, Y_test_new_list = [], []

    for cls in classes:
        cls_indices = np.where(Y_test == cls)[0]
        np.random.shuffle(cls_indices)

        split_idx = int(len(cls_indices) * val_ratio)
        val_indices = cls_indices[:split_idx]
        test_indices = cls_indices[split_idx:]

        X_val_list.append(X_test[val_indices])
        Y_val_list.append(Y_test[val_indices])
        X_test_new_list.append(X_test[test_indices])
        Y_test_new_list.append(Y_test[test_indices])

    X_val = np.concatenate(X_val_list, axis=0)
    Y_val = np.concatenate(Y_val_list, axis=0)
    X_test_new = np.concatenate(X_test_new_list, axis=0)
    Y_test_new = np.concatenate(Y_test_new_list, axis=0)

    if shuffle:
        val_perm = np.random.permutation(len(X_val))
        X_val, Y_val = X_val[val_perm], Y_val[val_perm]

        test_perm = np.random.permutation(len(X_test_new))
        X_test_new, Y_test_new = X_test_new[test_perm], Y_test_new[test_perm]

    return X_val, Y_val, X_test_new, Y_test_new