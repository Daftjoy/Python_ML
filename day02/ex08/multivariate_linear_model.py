import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../ex07")
from mylinearregression import MyLinearRegression as MyLR


def plot_model(x, y, data, theta, alpha = 0.001, max_iter = 100000):
    X = np.array(data[["{}".format(x)]])
    Y = np.array(data[["{}".format(y)]])
    myLR_obj = MyLR(theta, alpha, max_iter)
    print(myLR_obj.cost_(Y, myLR_obj.predict_(X)))
    myLR_obj.fit_(X[:,0].reshape(-1, 1), Y)
    Y_model = myLR_obj.predict_(X)
    print(myLR_obj.cost_(Y,Y_model))

    plt.plot(X,Y_model, 'gs')
    plt.plot(X,Y_model, 'g--', label = "Spredict(pills)")
    plt.plot(X, Y,'bo', label = "Strue")
    plt.grid(True)
    plt.legend(loc='upper right', bbox_to_anchor=(0.33, 1.15))
    plt.ylabel("Space driving score")
    plt.xlabel("Quantity of blue pill (in micrograms)")
    plt.show()






data = pd.read_csv("spacecraft_data.csv")
#plot_model("Age", "Sell_price", data, [[1000.0], [-1.0]], alpha = 2.5e-5, max_iter = 10000)
#plot_model("Thrust_power", "Sell_price", data, [[0], [10.0]], alpha = 1e-4, max_iter = 10000)
#plot_model("Terameters", "Sell_price", data, [[1000.0], [-1.0]], alpha = 1e-4, max_iter = 50000)
X = np.array(data[['Age','Thrust_power','Terameters']])
Y = np.array(data[["Sell_price"]])
myLR_obj = MyLR([1.0, 1.0, 1.0, 1.0], alpha = 5e-10, max_iter = 10000)
print(myLR_obj.cost_(Y, myLR_obj.predict_(X)))
myLR_obj.fit_(X, Y)
Y_model = myLR_obj.predict_(X)
print(myLR_obj.cost_(Y,Y_model))
plt.plot(X[:, 2],Y_model, 'go', label = "Spredict(pills)")
plt.plot(X[:, 2], Y,'bo', label = "Strue")
plt.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(0.33, 1.15))
plt.ylabel("Space driving score")
plt.xlabel("Quantity of blue pill (in micrograms)")
plt.show()

