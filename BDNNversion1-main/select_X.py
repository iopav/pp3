import numpy as np
def filter_data(X_train, Y_train, selection_percent):

    # Randomly select a percentage of the data
    n_samples_train = len(X_train)
    n_samples_to_select_train = int(selection_percent * n_samples_train)


    X_train_selected = X_train[:n_samples_to_select_train]
    Y_train_selected = Y_train[:n_samples_to_select_train]
    print('Y_train_selected.shape' + str(Y_train_selected.shape))
    return X_train_selected, Y_train_selected