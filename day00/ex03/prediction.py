import numpy as np

def simple_predict(x, theta):
    y_hat = []
    for i in x:
        y_hat.append(theta[0] + theta[1]*i)
    return(np.array(y_hat))

