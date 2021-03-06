import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("../ex09")
from my_logistic_regression import MyLogisticRegression

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
data = data_spliter(census, planets, 0.5)

#los 2 sets de datos, test y training, divididos en X e Y
xtr_set = data[0][0]
ytr_set = data[1][0]
xtest_set = data[0][1]
ytest_set = data[1][1]

#Los zipcodes convertidos a binario, es decir, son iguales a 0 o no
xtr_set = regularize(xtr_set)
#xtest_set =regularize(xtest_set)
for i in range (4):
    mylr = MyLogisticRegression([0, 0, 0, 0],alpha= 0.0001, n_cycle = 10000)
    mylr_t = MyLogisticRegression([0, 0, 0, 0],alpha= 0.001, n_cycle = 10000)
    mylr.fit_(xtr_set, np.array(ytest_set == i))
    if i == 0:
        pred = np.array(mylr.predict_(xtest_set).reshape(-1, 1))
        pred_t = np.array(mylr_t.predict_(xtest_set).reshape(-1, 1))
    else:
        pred = np.c_[pred,mylr.predict_(xtest_set).reshape(-1, 1)]
        pred_t = np.c_[pred,mylr_t.predict_(xtest_set).reshape(-1, 1)]

result = []
for val in pred:
    result.append(val.argmax())
result_t = []
for val in pred_t:
    result_t.append(val.argmax())
answer = np.array(result).reshape(-1, 1)
answer_t = np.array(result_t).reshape(-1, 1)
dif = 0
dif_t = 0
for i in range(answer.shape[0]):
    if answer[i] != ytest_set[i]:
        dif += 1
    if answer_t[i] != ytest_set[i]:
        dif_t += 1
print((dif/answer.shape[0])-(dif_t/answer.shape[0]))





