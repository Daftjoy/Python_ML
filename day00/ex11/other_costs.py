import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

def mse_(y, y_hat):
    return((y - y_hat).dot(y - y_hat) /len(y))

def rmse_(y, y_hat):
    return(sqrt((y - y_hat).dot(y - y_hat) /len(y)))

def mae_(y, y_hat):
    ret = 0
    for i in range(len(y)):
        ret +=  abs(y_hat[i] - y[i])
    return(float(ret/len(y_hat)))

def r2score_(y, y_hat):
    ret1 = 0
    ret2 = 0
    for i in range(len(y)):
        ret1 +=  (y_hat[i] - y[i])**2
        ret2 += (y_hat[i] - y.mean()) **2
    return(1 - (ret1/ret2))
