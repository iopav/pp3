


import numpy as np
import numpy.matlib as matlib

def update_list(arr, value, width):

    if isinstance(arr, list):
        return np.array(value, dtype=np.float64).reshape(1, 1)
    else:
        return np.concatenate(
            [np.array(arr, dtype=np.float64),
             matlib.repmat(value, 1, width)],
            axis=1
        )


