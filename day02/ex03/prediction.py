import numpy as np 

def add_intercept(x):
        arr = []
        if x.ndim == 1:
            for i in x:
                arr.append([1, i])
        else:
            for i in x:
                nl = []
                nl.append(1)
                for n in i:
                    nl.append(n)
                arr.append(nl)
        return(np.array(arr))

def predict_(x, theta):
    return(add_intercept(x).dot(theta))
