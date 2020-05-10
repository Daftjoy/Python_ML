import numpy as np 
import math

def add_intercept(x):
        if x.ndim == 1:
            x = np.insert(x, [0], 1, axis = 0)
        else: 
            x = np.insert(x, [0], 1, axis = 1)
        return(x)

def logistic_predict_(x, theta):
    x = add_intercept(x)
    if x.ndim == 0:
        return(np.array(np.array(1/(1 + math.exp(-(x * theta))))))
    exp = x.dot(theta)
    ret = []
    for i in exp:
        ret.append(np.array(1/(1 + math.exp(-i))))
    return(np.array(ret).reshape(-1,1))   
#con el (-1,1) dejamos en 1 el número de columnas\
# e infiere el número de filas (-1) del len 
