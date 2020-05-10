import numpy as np
import math

def minmax(x):
    arr = []
    for i in x:
        arr.append((i - np.amin(x)) / (np.amax(x) - np.amin(x)))
    return (np.array(arr))
