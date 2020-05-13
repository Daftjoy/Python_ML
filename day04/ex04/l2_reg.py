import numpy as np 

def iterative_l2(theta):
    sum = 0
    for i in range(1, theta.shape[0]):
        sum += theta[i]**2
    return (sum)

def l2(theta):
    theta[0] = 0
    return(np.sum(np.power(theta, 2)))





x = np.array([2, 14, -13, 5, 12, 4, -19])
y = np.array([3,0.5,-6])
print(l2(y))