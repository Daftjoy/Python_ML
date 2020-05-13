import numpy as np 

def l2(theta):
    theta[0] = 0
    return(np.sum(np.power(theta, 2)))

def reg_log_cost_(y, y_hat,theta, lambda_):
    eps = 1e-15
    ones = np.ones(y.shape[0]).reshape((-1,1))
    y = y.reshape(-1, 1)
    y_hat = y_hat.reshape(-1, 1)
    return(((-1/len(y)) * (np.dot(y.T, np.log(y_hat + eps))\
         + np.dot((ones - y).T, np.log(ones-(y_hat - eps)))))\
             + (lambda_*l2(theta)/(2*len(y))))


y = np.array([1, 1, 0, 0, 1, 1, 0])
y_hat = np.array([.9, .79, .12, .04, .89, .93, .01])
theta = np.array([1, 2.5, 1.5, -0.9])

print(reg_log_cost_(y, y_hat, theta, .05))