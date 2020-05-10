import numpy as np
import sys
sys.path.append("../ex00")
sys.path.append("../ex04")
sys.path.append("../ex01")
from prediction import predict_
from vec_gradient import gradient
from cost import cost_

def fit_(x, y, theta, alpha, max_iter):
    while cost_(y, predict_(x, theta)) != 0 and max_iter != 0:
        theta[0] = float(theta[0]- alpha * (gradient(x, y, theta)[0]))
        theta[1] = float(theta[1]- alpha * (gradient(x, y, theta)[1]))
        max_iter -= 1
    return(np.array(theta))
