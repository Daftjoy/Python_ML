import numpy as np 
import sys
sys.path.append("../ex00")
from prediction import predict_
def simple_gradient(x, y, theta1):
    ret = []
    s0 = 0
    s1 = 0
    y_hat = predict_(x, theta1)
    for i in range(len(y)):
        s0 += y_hat[i] - y[i]
        s1 += (y_hat[i]- y[i]) * x[i]
    ret.append(s0/len(y))
    ret.append(s1/len(y))
    return(np.array(ret))
