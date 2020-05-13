import numpy as np 
import sys 
sys.path.append("../ex04")
from log_pred import logistic_predict_
import math

def vec_log_loss_(y, y_hat, eps = 1e-15):
    ones = np.ones(y.shape[0]).reshape((-1,1))
    return(((-1/len(y)) * (np.dot(y.T, np.log(y_hat + eps))\
         + np.dot((ones - y).T, np.log(ones-(y_hat - eps)))))
