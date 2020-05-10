import numpy as np
import sys
sys.path.append("../ex04")
from log_pred import logistic_predict_

def log_gradient(x, y, theta):
    y_hat = logistic_predict_(x, theta)
    sum = []
    for t in range(theta.shape[0]):
        sum.append(0)
    for i in range(len(y)):
        for j in range(len(theta)):
            if j == 0:
                sum[j] += (y_hat[i] - y[i])/len(y)
            else:
                if (x[i].size > 1):
                    sum[j] += ((y_hat[i] - y[i]) * x[i][j -1])/len(y)
                else:
                    sum[j] += ((y_hat[i] - y[i]) * x[i])/len(y)
    return(sum)

