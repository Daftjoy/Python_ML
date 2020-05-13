import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import sys
sys.path.append("../ex11")
from my_logistic_regression import MyLogisticRegression as MyLR
sys.path.append("../ex03")
from polynomial_model_extended import add_polynomial_features as adp
sys.path.append("../../day03/ex11")
from other_metrics import *
def regularize(x):
    for i in range(x.shape[1]):
        xtr_set[i,:] =(xtr_set[i,:] -(sum(xtr_set[i,:])/xtr_set[i,:].shape[0]))/(np.amax(xtr_set[i,:]) - np.amin(xtr_set[i,:]))
    return (x)

def data_spliter(x, y, proportion):
    indexes = np.arange(y.shape[0])
    np.random.shuffle(indexes)
    return(np.split(x[indexes], [int(x.shape[0] * proportion)]),\
        np.split(y[indexes], [int(y.shape[0] * proportion)]))

dcensus = pd.read_csv("solar_system_census.csv")
dplanets = pd.read_csv("solar_system_census_planets.csv")

census = np.array(dcensus[["height", "weight", "bone_density"]]).reshape(-1,3)
planets = np.array(dplanets[["Origin"]]).reshape(-1,1)
data = data_spliter(census, planets, 0.7)

#los 2 sets de datos, test y training, divididos en X e Y
xtr_set = adp(data[0][0], 3)
ytr_set = data[1][0]
xtest_set = adp(data[0][1], 3)
ytest_set = data[1][1]
theta = np.zeros(xtr_set.shape[1] + 1)
#Los zipcodes convertidos a binario, es decir, son iguales a 0 o no
xtr_set = regularize(xtr_set)
#xtest_set =regularize(xtest_set)
for i in range (4):
    mylr = MyLR(theta,alpha= 0.0001, n_cycle = 100, penalty = 1)
    mylr.fit_(xtr_set, np.array(ytr_set == i))
    if i == 0:
        pred = np.array(mylr.predict_(xtest_set).reshape(-1, 1))
    else:
        pred = np.c_[pred,mylr.predict_(xtest_set).reshape(-1, 1)]

result = []
n_answer = []
for val in pred:
    result.append(val.argmax())
answer = np.array(result).reshape(-1, 1)
for j in range(pred.shape[1]):
    for i in range(pred.shape[0]):
        if pred[i][j] > 0.01:
            n_answer.append(1)
        else:
            n_answer.append(0)
    arr_answer = np.array(n_answer).reshape(-1, 1)
    print(f1_score_(np.array(ytest_set == j), arr_answer, pos_label = 1))
#print(f1_score_(ytest_set, answer, pos_label = 1))

