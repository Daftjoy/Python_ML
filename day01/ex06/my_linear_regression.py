import numpy as np

class MyLinearRegression():
    def __init__(self,  thetas, alpha=0.001, max_iter=100000):
              self.alpha = alpha
              self.max_iter = max_iter
              self.thetas = thetas
    
    def add_intercept(self, x):
        arr = []
        if x.ndim == 1:
            for i in x:
                arr.append([1, i])
        else:
            for i in x:
                nl = []
                nl.append(1)
                for n in i:
                    nl.append(n)
                arr.append(nl)
        return(np.array(arr))

    def predict_(self,x):
        return(self.add_intercept(x).dot(self.thetas))
    
    def cost_elem_(self, y, y_hat):
        ret = []
        y = y.reshape(-1, 1)
        for i in range(len(y)):
            ret.append((1/(2*len(y))) * ((y_hat[i] - y[i])**2))
        return(np.array(ret))

    def cost_(self, y, y_hat):
        y = y.reshape(-1, 1)
        y_hat = y_hat.reshape(-1, 1)
        return((y - y_hat).T.dot(y - y_hat) /(len(y) * 2))
    
    def fit_(self, x, y):
        while self.cost_(y, self.predict_(x)) != 0 and self.max_iter != 0:
            self.thetas[0] = float(self.thetas[0]- self.alpha * (self.gradient(x, (y), self.thetas)[0]))
            self.thetas[1] = float(self.thetas[1]- self.alpha * (self.gradient(x, (y), self.thetas)[1]))
            self.max_iter -= 1
        return(self.thetas)
    
    def gradient(self, x, y, theta):
        xt = self.add_intercept(x)
        y_hat = self.predict_(x)
        return(((np.transpose(xt)).dot(y_hat -y))/len(y))
