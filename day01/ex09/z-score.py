import numpy as np
import math

def std(x):
        err = 0
        sum = 0
        for i in x:
            sum += i
        mean =sum/len(x)
        for i in x:
            err += (i - mean)**2
        return(math.sqrt(err/len(x)))

def mean(x):
        sum = 0
        for i in x:
            sum += i
        return(sum/len(x))

def zscore(x):
    arr = []
    for i in x:
        arr.append((i - mean(x)) / std(x))
    return (np.array(arr))



