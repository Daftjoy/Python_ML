import numpy as np 

def add_intercept(x):
        if x.ndim == 1:
            x = np.insert(x, [0], 1, axis = 0)
        else: 
            x = np.insert(x, [0], 1, axis = 1)
        return(x)

def logistic_predict_(x, theta):
    x = add_intercept(x)
    if x.ndim == 0:
        return(np.array(np.array(1/(1 + np.exp(-(x * theta))))))
    exp = x.dot(theta)
    ret = []
    for i in exp:
        ret.append(np.array(1/(1 + np.exp(-i))))
    return(np.array(ret).reshape(-1,1))   
#con el (-1,1) dejamos en 1 el número de columnas\
# e infiere el número de filas (-1) del len 


def reg_logistic_grad(y, x, theta, lambda_):
    y_hat = logistic_predict_(x, theta)
    sum = np.zeros((theta.shape[0], 1))
    for i in range(y.shape[0]):
        sum[0] += y_hat[i] - y[i]
        for j in range(1, len(theta)):
            if (x[i].size > 1):
                sum[j] += (y_hat[i] - y[i]) * x[i][j -1]
            else:
                sum[j] += (y_hat[i] - y[i]) * x[i]
    for j in range(1, len(theta)):
        sum[j] += lambda_ *theta[j] 
    return(sum/len(y))

def vec_reg_logistic_grad(y, x, theta, lambda_):
    y_hat = logistic_predict_(x, theta)
    xT = add_intercept(x).reshape(-1, x[0].size + 1).T 
    theta[0] = 0
    return((xT.dot((y_hat - y)) + (lambda_ * theta))/ len(y))



