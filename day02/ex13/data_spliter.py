import numpy as np 

def data_spliter(x, y, proportion):
    indexes = np.arange(y.shape[0])
    np.random.shuffle(indexes)
    return(np.split(x[indexes], [int(x.shape[0] * proportion)]),\
        np.split(y[indexes], [int(y.shape[0] * proportion)]))
