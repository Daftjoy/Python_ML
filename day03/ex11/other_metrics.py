import numpy  as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def metrics(y, y_hat, pos_label = 1):
    met = []
    for i in range(4):
        met.append(0)
    for i in range(y.shape[0]):
        if y[i] != y_hat[i]:
            if y_hat[i] != pos_label:
                met[3] += 1
            else:
                met[2] += 1
        else:
            if y_hat[i] != pos_label:
                met[1] += 1
            else:
                met[0] += 1
    return(met)
    """
    0 = tp, 1 = tn, 2= fp, 3 = fn
    when used for accuracy, individual variables(tp, tn...) don't match exactly
    the formula given, but as you only need right and wrong class it still works
    """

def accuracy_score_(y, y_hat):
    tp, tn, fp, fn = metrics(y,y_hat)
    return((tp + tn)/(tp + fp + tn + fn))
    #that's the real formula, wrong one in subject

def precision_score_(y, y_hat, pos_label=1):
    tp, fp = metrics(y,y_hat, pos_label)[0], metrics(y,y_hat, pos_label)[2]
    return(tp/(tp + fp))

def recall_score_(y, y_hat, pos_label=1):
    tp, fn = metrics(y,y_hat, pos_label)[0], metrics(y,y_hat, pos_label)[3]
    return(tp/(tp + fn))

def f1_score_(y, y_hat, pos_label=1):
    return(2 * precision_score_(y, y_hat, pos_label) * recall_score_(y, y_hat, pos_label)\
    /(precision_score_(y, y_hat, pos_label) + recall_score_(y, y_hat, pos_label)))

