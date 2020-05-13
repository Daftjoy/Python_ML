import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import sys
sys.path.append("../ex09")
from ridge import MyRidge as MyR
sys.path.append("../ex03")
from polynomial_model_extended import add_polynomial_features as adp

def data_spliter(x, y, proportion):
    indexes = np.arange(y.shape[0])
    np.random.shuffle(indexes)
    return(np.split(x[indexes], [int(x.shape[0] * proportion)]),\
        np.split(y[indexes], [int(y.shape[0] * proportion)]))

def mse_(y, y_hat):
    return((y - y_hat).T.dot(y - y_hat) /len(y))

data =  pd.read_csv("spacecraft_data.csv")
X = np.array(data[['Age','Thrust_power','Terameters']]).reshape(-1, 3)
Y = np.array(data[["Sell_price"]]).reshape(-1, 1)

n_data = data_spliter(X, Y, 0.6)


xtr_set = adp(n_data[0][0], 3)
ytr_set = n_data[1][0]
xtest_set = adp(n_data[0][1], 3)
ytest_set = n_data[1][1]
thetas = np.ones(xtr_set.shape[1] + 1).reshape((-1,1))
l = 0
larr = []
msearr =[]
while l <= 2:
    myLR_obj = MyR(thetas, alpha = 5e-16, n_cycle = 100, lambda_= l)
    myLR_obj.fit_(xtr_set, ytr_set)
    larr.append(l)
    """
    plt.plot(xtest_set[:,0],myLR_obj.predict_(xtest_set), 'go--', label = "Prediction")
    plt.plot(xtest_set[:,0],ytest_set,'bo')
    plt.show()
    """
    msearr.append(float(mse_(ytest_set, myLR_obj.predict_(xtest_set))))
    l += 0.1
plt.bar(larr, msearr, width =0.02, edgecolor = 'r')
plt.show()