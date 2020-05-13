import numpy as np 


def l2(theta):
    theta[0] = 0
    return(np.sum(np.power(theta, 2)))

def reg_cost_(y, y_hat, theta, lambda_):
    return(((y - y_hat).dot(y - y_hat) + lambda_*l2(theta))/(len(y) * 2))
