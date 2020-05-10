import numpy as np
def cost_(y, y_hat):
    ret = (y - y_hat).dot(y - y_hat) /(2 * len(y))
    return (ret)

