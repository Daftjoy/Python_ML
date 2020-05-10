import numpy as np 
import sys
sys.path.append("../ex00")
sys.path.append("../../day00/ex04")
from prediction import predict_
from tools import add_intercept

def gradient(x, y, theta):
    xt = add_intercept(x)
    y_hat = predict_(x, theta)
    return((y_hat -y).dot(xt)/len(y))

