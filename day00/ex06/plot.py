import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../ex05")
from prediction import predict_

def plot(x, y, theta):
    for i in range(len(x)):
        plt.plot(x[i],y[i],'bo')
    plt.plot(x,predict_(x,theta), 'r')
    plt.show()
