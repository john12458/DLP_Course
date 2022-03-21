import numpy as np
""" cross entropy v2"""
def cross_Entropy2(y, labels):
    m = labels.shape[1]   
    cost = (1./m) * (-np.dot(labels,np.log(y).T) - np.dot(1-labels, np.log(1-y).T))
    cost = np.squeeze(cost)  # makes sure cost is the dimension we expect.
    return cost
def derivative_cross_Entropy2(y, labels):
    a = np.divide(labels,y +1e-10)
    b = np.divide(1-labels, 1-y +1e-10)
    dc = -(a - b)
    return dc
""" binary cross entropy"""
def cross_Entropy(y, labels):
    cost = -np.log((1-labels) + ((2*labels-1)*y))
    return cost.mean()
      # if labels == 1:
    #   return -np.log(y)
    # else:
    #   return -np.log(1 - y)
def derivative_cross_Entropy(y, labels):
    dc = (1-2*labels)/((1-labels) + ((2*labels-1)*y))
    # print(dc)
    return dc
""" mse """
def mse(x,y):
    return (((x-y)**2)*0.5).mean()
def derivative_mse(x,y):
    return (x-y)