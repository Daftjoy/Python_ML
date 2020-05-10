import numpy as np 

def simple_predict(x, theta):
    arr = []
    for i in x:
        res = theta[0]
        for j in range(len(i)):
            res += theta[j + 1] * i[j]
        arr.append(res)
    return(np.array(arr))
