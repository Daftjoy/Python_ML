import numpy as np 
import sys 
sys.path.append("../ex03")
from prediction import predict_

def cost_(y, y_hat):
    return((y - y_hat).dot(y - y_hat) /(len(y) * 2))
