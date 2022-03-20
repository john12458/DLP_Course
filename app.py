import numpy as np
import matplotlib.pyplot as plt
def generate_linear(n=100):
    pts = np.random.uniform(0,1,(n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0],pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs),np.array(labels).reshape(n,1)
def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i,0.1*i])
        labels.append(0)
        if 0.1 * i == 0.5:
            continue
        inputs.append([0.1*i,1-0.1*i])
        labels.append(1)
    return np.array(inputs),np.array(labels).reshape(21,1)
class Linear():
    def __init__(self,in_c,out_c):
        self.weight = np.random.randn(out_c,in_c) 
        self.bias = np.zeros([out_c, 1])
        self.x = None
        self.y = None
    def forward(self,x):
        self.x = x
        self.y = np.dot(self.weight, self.x) + self.bias  # x = [21,2], W = [2,1] ,y = [21,1]
        return self.y
    def backward(self,grad,lr):
        #z2 = a1w2 --> dw2 = a1.T @ G
        new_grad = np.dot(self.weight.T, grad)
        grad_for_weight = np.dot(grad,self.x.T)
        grad_for_bias = np.sum(grad,axis=1,keepdims=True)
        # print("grad_for_weight ",grad_for_weight )
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
# def Cross_Entropy(y, labels):
#     m = labels.shape[1]   
#     cost = (1./m) * (-np.dot(labels,np.log(y).T) - np.dot(1-labels, np.log(1-y).T))
#     cost = np.squeeze(cost)  # makes sure cost is the dimension we expect.
#     return cost
# def derivative_Cross_Entropy(y, labels):
#     a = np.divide(labels,y +1e-10)
#     # print("a",a)
#     b = np.divide(1-labels, 1-y +1e-10)
#     # print("b",b)
#     dc = -(a - b)
#     return dc
def Cross_Entropy(y, labels):
    cost = -np.log((1-labels) + ((2*labels-1)*y))
    # print(cost)
    return cost.mean()
    # if labels == 1:
    #   return -np.log(y)
    # else:
    #   return -np.log(1 - y)

def derivative_Cross_Entropy(y, labels):
    dc = (1-2*labels)/((1-labels) + ((2*labels-1)*y))
    # print(dc)
    return dc
def mse(x,y):
    return (((x-y)**2)*0.5).mean()
def derivative_mse(x,y):
    return (x-y)
def train(x, labels,fun_list,lr):
    """ forward """
    y = x.copy()
    # print(fun_list)
    # cnt = 0
    for f in fun_list:
        # if cnt == 0:
        #     # print("self.weight",f.weight)
        #     cnt = 1
        y = f.forward(y)

    loss = mse(y,labels)
    """backward"""
    grad = derivative_mse(y,labels)
    # print(grad)
    fun_list.reverse()
    for f in fun_list:
        grad = f.backward(grad,lr)
        # print(grad)
    # print("------")
    fun_list.reverse()

    

    return loss,y
def plot_data(data,labels):
    color_label = ['red' if i==0 else 'blue' for i in labels.squeeze()]
    plt.scatter(data[:,0],data[:,1],c= color_label)
def show_results(x,labels,pred_y):
    plt.subplot(1,2,1)
    plt.title("Ground truth",fontsize=18)
    for i in range(x.shape[0]):
        if labels.T[i] == 0:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
    plt.subplot(1,2,2) 
    plt.title("predict results",fontsize=18)
    for i in range(x.shape[0]):
        if pred_y.T[i] == 0:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
    plt.show()
if __name__ == '__main__':
    np.random.seed(5)
    # data,labels = generate_linear(n=100)
    data,labels = generate_XOR_easy() 
    x = data.T
    labels = labels.T
    channels = [2,6,6,1]
    fun_list = []
    for idx,(in_c,out_c) in enumerate(zip(channels[:-1],channels[1:])):
        fun_list.append(Linear(in_c,out_c))
        if idx == len(channels)-2:
            fun_list.append(Sigmoid())

        else:
            fun_list.append(Relu())
    steps = 10000
    lr = 0.0003
    loss_list = []
    pred_y = []
    for step in range(steps):
        loss,y = train(x, labels,fun_list,lr)
        loss_list.append(loss)
        pred_y.append(y)
    print("labels",labels.squeeze()[0])
    print("pred_y",pred_y[0].squeeze()[0])
    print("pred_y",pred_y[-1].squeeze()[0])
    pred_y = pred_y[-1]
    print(pred_y)
    plt.plot(loss_list)
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
    # print(1.0*(pred_y>0.48))
    show_results(data,labels,1.0*(pred_y>0.5))


