import numpy as np 
import sys
sys.path.append("../ex03")
from prediction import *

def gradient(x, y, theta):
    return((np.transpose(add_intercept(x))).dot(predict_(x, theta) - y)/ len(y))
