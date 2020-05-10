import numpy as np

class MyLinearRegression():
    """
    Description:
        My personnal linear regression class to fit like a boss.
    """
    def __init__(self,  thetas, alpha=0.001, max_iter=100000):
              self.alpha = alpha
              self.max_iter = max_iter
              self.thetas = thetas
    
    def add_intercept(self, x):
        if x.ndim == 1:
            x = np.insert(x, [0], 1, axis = 0)
        else: 
            x = np.insert(x, [0], 1, axis = 1)
        return(x)

    def predict_(self,x):
        return(self.add_intercept(x).dot(self.thetas))
    
    def cost_elem_(self, y, y_hat):
        ret = []
        y = np.concatenate(y)
        for i in range(len(y)):
            ret.append((1/(2*len(y))) * ((y_hat[i] - y[i])**2))
        return(np.array(ret))

    def cost_(self, y, y_hat):
        #y = np.concatenate(y)
        #y_hat = np.concatenate(y_hat)
        return(np.transpose((y - y_hat)).dot(y - y_hat) /(len(y) * 2))
    
    def fit_(self, x, y):
        n_cycles =  self.max_iter
        while n_cycles != 0:
            for n in range(len(self.thetas)):
                self.thetas[n] = self.thetas[n]- self.alpha * np.sum((self.gradient(x,y, self.thetas)[n,:]))
            n_cycles -= 1
        return(self.thetas)
    
    def gradient(self, x, y, theta):
        xt = self.add_intercept(x)
        y_hat = self.predict_(x)
        return(((np.transpose(xt)).dot(y_hat -y))/len(y))
