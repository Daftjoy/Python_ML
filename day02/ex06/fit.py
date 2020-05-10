import numpy as np 
import sys
sys.path.append("../ex04")
sys.path.append("../ex03")
sys.path.append("../ex05")
from cost import cost_
from prediction import *
from gradient import gradient

def fit_(x, y, theta, alpha, n_cycles):
    while n_cycles != 0:
        for n in range(len(theta)):
            theta[n] = float(theta[n]- alpha * (gradient(x, (y), theta)[n]))
        n_cycles -= 1
    return(theta)