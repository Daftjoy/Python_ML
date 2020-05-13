import numpy as np 
def add_intercept(x):
    if x.ndim == 1:
        x = np.insert(x, [0], 1, axis = 0)
    else: 
        x = np.insert(x, [0], 1, axis = 1)
    return(x)

def predict_(x, theta):
    return(add_intercept(x).dot(theta))

def reg_linear_grad(y, x, theta, lambda_):
    res = np.zeros((len(theta), 1))
    for i in range(y.shape[0]):
        res[0] += predict_(x[i],theta) - y[i]
        for j in range(1, len(theta)):
            res[j] += (predict_(x[i],theta) - y[i])*x[i][j-1]
    for j in range(1, len(theta)):
        res[j] += lambda_*theta[j]
    return(res/y.shape[0])

def vec_reg_linear_grad(y, x, theta, lambda_):
    xt = add_intercept(x)
    y_hat = predict_(x, theta)
    theta[0] = 0
    return(((np.transpose(xt)).dot(y_hat -y) + (lambda_ * theta))/len(y))
