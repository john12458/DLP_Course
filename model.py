import numpy as np

class Linear():
    def __init__(self,in_c,out_c):
        self.weight = np.random.randn(in_c,out_c) 
        self.bias = np.zeros([1,out_c])
        self.x = None
        self.y = None
    def forward(self,x):
        self.x = x
        self.y = np.dot(self.x,self.weight) + self.bias  # x = [21,2], W = [2,1] ,y = [21,1]
        return self.y
    def backward(self,grad,lr):
        """ calculate gradient """
        new_grad = np.dot(grad,self.weight.T) # y = xw --> dL/dx = G @ W.T , G = dL/dy
        grad_for_weight = np.dot(self.x.T,grad) # y = xw --> dL/dw = x.T @ G , G = dL/dy
        grad_for_bias = np.sum(grad,axis=1,keepdims=True)
        """ update weight and bias """
        self.weight = self.weight - lr*grad_for_weight
        self.bias = self.bias - lr*grad_for_bias
        return new_grad

class Sigmoid():
    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))  
    def derivative_sigmoid(self,x):
        return x * (1.0-x)
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self,x):
        self.x = x
        self.y = self.sigmoid(x)
        return self.y
    def backward(self,grad,*kwargs):
        new_grad = grad * self.derivative_sigmoid(self.y)
        return new_grad

class Relu():
    def relu(self,x):
        return np.maximum(0,x) 
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self,x):
        self.x = x
        self.y = self.relu(x)
        return self.y
    def backward(self,grad,*kwargs):
        new_grad = np.array(grad,copy=True)  # a0 = relu(z0)
        new_grad[self.x<=0] = 0
        return new_grad