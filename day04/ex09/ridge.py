import numpy as np 

class MyRidge():
    def __init__(self, thetas, alpha=0.001, n_cycle=1000, lambda_=0.5):
        self.alpha = alpha
        self.max_iter = n_cycle
        self.thetas = thetas
        self.lambda_ = lambda_

    def add_intercept(self, x):
        if x.ndim == 1:
            x = np.insert(x, [0], 1, axis = 0)
        else: 
            x = np.insert(x, [0], 1, axis = 1)
        return(x)

    def predict_(self, x):
        return((self.add_intercept(x).dot(self.thetas)).reshape(-1, 1))

    def l2(self):
        theta = self.thetas
        theta[0] = 0
        return(np.sum(np.power(theta, 2)))

    def cost_(self, y, y_hat):
        return(((y - y_hat).T.dot(y - y_hat) + \
            self.lambda_*self.l2())/(len(y) * 2))

    def gradient(self, x, y):
        theta = np.array(self.thetas)
        xt = self.add_intercept(x)
        y_hat = self.predict_(x)
        theta[0] = 0
        return(((np.transpose(xt)).dot(y_hat -y) + (self.lambda_ * theta))/len(y))

    def fit_(self, x, y):
        n_cycles =  self.max_iter
        while n_cycles != 0:
            for n in range(len(self.thetas)):
                self.thetas[n] = self.thetas[n]- self.alpha * np.sum((self.gradient(x,y)[n,:]))
            n_cycles -= 1
        return(self.thetas)

    