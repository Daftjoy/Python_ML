import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys 
sys.path.append("../ex07")
from mylinearregression import MyLinearRegression as MyLR
sys.path.append("../ex10")
from polinomial_model import add_polynomial_features




data = pd.read_csv("are_blue_pills_magics.csv")
X = np.array(data["Micrograms"]).reshape(-1,1)
Y = np.array(data["Score"]).reshape(-1,1)


myLR_obj = MyLR([90.0, -1.0, 1.0, 2.0], alpha = 5e-6, max_iter = 20000)
print(myLR_obj.cost_(Y, add_polynomial_features(X, 3)))
myLR_obj.fit_(add_polynomial_features(X, 3), Y)
Y_model = myLR_obj.predict_(add_polynomial_features(X, 3))
print(myLR_obj.cost_(Y,Y_model))

plt.plot(X,Y_model, 'go', label = "Spredict(pills)")
plt.plot(X, Y,'bo', label = "Strue")
plt.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(0.33, 1.15))
plt.ylabel("Space driving score")
plt.xlabel("Quantity of blue pill (in micrograms)")
plt.show()
