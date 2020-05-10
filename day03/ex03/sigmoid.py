import numpy as np 
import math

def sigmoid_(x):
    if x.ndim == 0:
        return(np.array(np.array(1/(1 + math.exp(x)))))
    ret = []
    for i in x:
        ret.append(np.array(1/(1 + math.exp(-i))))
    return(np.array(ret))
