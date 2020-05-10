import numpy as np 
import sys
sys.path.append("../ex04")
from log_pred import *

def vec_log_gradient(x, y, theta):
    #x[0].size + 1 arregla problemas con arrays de varias dimensiones
    #de esa forma ajusta el tamaó de xT al número de xn para cada y (+ 1)
    xT = add_intercept(x).reshape(-1, x[0].size + 1).T 
    return(xT.dot(logistic_predict_(x, theta) - y) / len(y))
