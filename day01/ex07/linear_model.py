import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../ex06")
sys.path.append("../resources")
from sklearn.metrics import mean_squared_error
from my_linear_regression import MyLinearRegression as MyLR

def print_costfn(t0, y):
    for i in np.linspace(t0 -10, t0 +50, 3000):
        linear_model3 = MyLR(np.array([[-10], [i]]))
        Y_model3 = linear_model3.predict_(Xpill)
        plt.plot(linear_model3.thetas[1], linear_model3.cost_(y, Y_model3), 'gs')




data = pd.read_csv("are_blue_pills_magics.csv")
Xpill = np.array(data["Micrograms"]).reshape(-1,1)
Yscore = np.array(data["Score"]).reshape(-1,1)
linear_model1 = MyLR(np.array([[89.0], [-8]]))
linear_model2 = MyLR(np.array([[89.0], [-6]]))
Y_model1 = linear_model1.predict_(Xpill)
Y_model2 = linear_model2.predict_(Xpill)

linear_model1_2 = MyLR(linear_model1.fit_(Xpill, Yscore))
Y_model1_2 = linear_model1_2.predict_(Xpill)

print(linear_model1.cost_(Yscore, Y_model1) * 2)
print(mean_squared_error(Yscore, Y_model1))
print(linear_model1.cost_(Yscore, Y_model2) * 2)
print(mean_squared_error(Yscore, Y_model2))


plt.plot(Xpill,Y_model1_2, 'gs')
plt.plot(Xpill,Y_model1_2, 'g--', label = "Spredict(pills)")
plt.plot(Xpill,Yscore,'bo', label = "Strue")
plt.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(0.33, 1.15))
plt.ylabel("Space driving score")
plt.xlabel("Quantity of blue pill (in micrograms)")
plt.show()

for t0 in range(-10,-5, 1):
   print_costfn(t0, Yscore)
plt.show()