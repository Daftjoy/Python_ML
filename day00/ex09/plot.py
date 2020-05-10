import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../ex05")
sys.path.append("../ex08")
from prediction import predict_
from vec_cost import cost_

def plot_with_cost(x, y, theta):
    for i in range(len(x)):
        plt.plot(x[i],y[i],'bo')
    plt.plot(x,predict_(x,theta), 'r')
    plt.vlines(x, y, predict_(x, theta), 'r', "dashed")
    plt.title("Cost: {}".format(cost_(y,predict_(x,theta))))
    plt.show()
