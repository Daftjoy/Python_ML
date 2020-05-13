import numpy as np 

def add_polynomial_features(x, power):

    x_aux = x
    n_c = x.shape[1]
    for p in range(2, power + 1):
        for i in range(n_c):
            x = np.insert(x, [(n_c * (p-1)) + i], np.power(x_aux[:,i], p).reshape(-1, 1), axis = 1)
    return(np.array(x))

