import numpy as np
import sys
sys.path.append("../../day00/ex04")
from tools import add_intercept

def predict_(x, theta):
    return(add_intercept(x).dot(theta))
