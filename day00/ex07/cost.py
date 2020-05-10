import numpy as np 
import sys
sys.path.append("../ex05")
from prediction import predict_

def cost_elem_(y, y_hat):
    ret = []
    for i in range(len(y)):
        ret.append((1/(2*len(y))) * (y_hat[i] - y[i])**2)
    return(np.array(ret))

def cost_(y, y_hat):
    ret = 0
    for i in range(len(y)):
        ret +=  (y_hat[i] - y[i])**2
    return(float(ret/(2*len(y_hat))))
