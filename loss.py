import numpy as np
from util import cross_Entropy2,derivative_cross_Entropy2,cross_Entropy,derivative_cross_Entropy,mse,derivative_mse

class Cross_Entropy():
    def __init__(self):
        self.y = None
        self.labels = None
    def forward(self,y,labels):
        self.y = y
        self.labels = labels
        return cross_Entropy(self.y, self.labels)
    def backward(self):
        return derivative_cross_Entropy(self.y,self.labels)

class MSE():
    def __init__(self):
        self.y = None
        self.labels = None
    def forward(self,y,labels):
        self.y = y
        self.labels = labels
        return mse(self.y, self.labels)
    def backward(self):
        return derivative_mse(self.y,self.labels)
