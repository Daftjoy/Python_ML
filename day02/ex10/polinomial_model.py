import numpy as np 

def add_polynomial_features(x, power):
    x_aux = x
    if power == 2:
        x = np.insert(x, [1], x_aux**2, axis = 1)
        return(x)
    for p in range(2, power + 1):
        x = np.insert(x, [p-1], x_aux**p, axis = 1)
    return(x)
