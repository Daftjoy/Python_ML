import numpy as np 
import math
class MyLogisticRegression():
    def __init__(self, theta, alpha=0.001, n_cycle=100000):
        self.alpha = alpha
        self.max_iter = n_cycle
        self.theta = theta
    
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
            ret.append(np.array(1/(1 + math.exp(-i))))
        return(np.array(ret).reshape(-1,1))   

    def cost_(self, x, y):
        eps = float(1e-15)
        y_hat = self.predict_(x)
        sum = 0
        for i in range(len(y)):
            sum += (y[i] * math.log(y_hat[i] + eps)) + (1 - y[i]) * math.log(1-(y_hat[i] - eps))
        return(sum * (-1/len(y)))

    def gradient(self, x, y):
    #x[0].size + 1 arregla problemas con arrays de varias dimensiones
    #de esa forma ajusta el tamaó de xT al número de xn para cada y (+ 1)
        xT = self.add_intercept(x).reshape(-1, x[0].size + 1).T 
        return(xT.dot(self.predict_(x) - y) / len(y))

    def fit_(self, x, y):
        n_cycles =  self.max_iter
        while n_cycles != 0:
            for n in range(len(self.theta)):
                self.theta[n] = self.theta[n]- self.alpha * np.sum((self.gradient(x,y)[n,:]))
            n_cycles -= 1
        return(self.theta)

