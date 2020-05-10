import numpy as np 

def confusion_matrix_(y_true, y_hat, labels=None):
    if labels is None:
        labels = []
        for i in range(y_true.shape[0]):
            if y_true[i] not in labels:
                labels.append(y_true[i])
            elif y_hat[i] not in labels:
                labels.append(y_hat[i])
    labels.sort()
    ret = np.zeros((len(labels), len(labels)))
    for i in range(y_true.shape[0]):
        if y_hat[i] in labels and y_true[i] in labels:
            ret[labels.index("{}".format(y_true[i]))]\
                [labels.index("{}".format(y_hat[i]))] += 1
    return(ret)
