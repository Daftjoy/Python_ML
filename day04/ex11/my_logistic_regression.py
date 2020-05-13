import numpy as np 
import math
class MyLogisticRegression():
    def __init__(self, theta, alpha=0.001, n_cycle=100000, penalty = 'l2'):
        self.alpha = alpha
        self.max_iter = n_cycle
        self.theta = theta
        self.lambda_ = penalty
    
    def l2(self):
        thetas = np.array(self.theta)
        thetas[0] = 0
        return(np.sum(np.power(thetas, 2)))


    def add_intercept(self, x):
        if x.ndim == 1:
            x = np.insert(x, [0], 1, axis = 0)
        else: 
            x = np.insert(x, [0], 1, axis = 1)
        return(x)
    
    def predict_(self, x):
        x = self.add_intercept(x)
        if x.ndim == 0:
            return(np.array(np.array(1/(1 + math.exp(-(x * self.theta))))))
        exp = x.dot(self.theta)
        ret = []
        for i in exp:
            ret.append(np.array(1/(1 + np.exp(-i))))
        return(np.array(ret).reshape(-1,1))   

    def cost_(self, y, y_hat):
        eps = 1e-15
        ones = np.ones(y.shape[0]).reshape((-1,1))
        y = y.reshape(-1, 1)
        y_hat = y_hat.reshape(-1, 1)
        return(((-1/len(y)) * (np.dot(y.T, np.log(y_hat + eps))\
             + np.dot((ones - y).T, np.log(ones-(y_hat - eps)))))\
                 + (self.lambda_*self.l2()/(2*len(y))))

    def gradient(self, x, y):
        thetas = np.array(self.theta)
        y_hat = self.predict_(x)
        xT = self.add_intercept(x).reshape(-1, x[0].size + 1).T 
        thetas[0] = 0
        return((xT.dot((y_hat - y)) + (self.lambda_ * thetas))/ len(y))

    def fit_(self, x, y):
        n_cycles =  self.max_iter
        while n_cycles != 0:
            for n in range(len(self.theta)):
                self.theta[n] = self.theta[n]- self.alpha * np.sum((self.gradient(x,y)[n,:]))
            n_cycles -= 1
        return(self.theta)