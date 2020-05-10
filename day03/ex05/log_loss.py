import numpy as np 
import sys 
sys.path.append("../ex04")
from log_pred import logistic_predict_
import math

def log_loss_(y, y_hat, eps = 1e-15):
    sum = 0
    for i in range(len(y)):
        sum += (y[i] * math.log(y_hat[i] + eps)) + (1 - y[i]) * math.log(1-(y_hat[i] + eps))
    return(sum * (-1/len(y)))
