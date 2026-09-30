

import numpy as np

def standardize_matrix(matrix):
    min_vals = np.min(matrix, axis=0)
    max_vals = np.max(matrix, axis=0)
    standardized_matrix = (matrix - min_vals) / (max_vals - min_vals)

    return standardized_matrix



def softmax_standardization(matrix):
    exp_matrix = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
    softmax_matrix = exp_matrix / np.sum(exp_matrix, axis=1, keepdims=True)
    #print('row in softmax_matrix is: ' + str(softmax_matrix[1,:]))
    return softmax_matrix

#def max_standardization(matrix):

