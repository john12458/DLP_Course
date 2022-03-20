from dis import pretty_flags
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

def plot_data(data,labels,show=False):
    # plt.figure()
    color_label = ['red' if i==0 else 'blue' for i in labels]
    plt.scatter(data[:,0],data[:,1],c= color_label)
    # if show:
    #     plt.show()

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))  
def derivative_sigmoid(x):
    return x * (1.0-x)

def relu(x):
    return np.maximum(0,x)

def mse(x,y):
    return ((x-y)**2)*0.5
def derivative_mse(x,y):
    return (x-y)
    # return 2*(x-y)


# def Cross_Entropy(y, labels):
#     cost = -np.log((1-labels) + ((2*labels-1)*y))
#     # print(cost)
#     return cost
#     # if labels == 1:
#     #   return -np.log(y)
#     # else:
#     #   return -np.log(1 - y)

# def derivative_Cross_Entropy(y, labels):
#     dc = (1-2*labels)/((1-labels) + ((2*labels-1)*y))
#     # print(dc)
#     return dc

def Cross_Entropy(y, labels):
    y = y.reshape(1,-1)
    labels = labels.reshape(1,-1)
    m = labels.shape[1]   
    cost = (1./m) * (-np.dot(labels,np.log(y).T) - np.dot(1-labels, np.log(1-y).T))
    cost = np.squeeze(cost)  # makes sure cost is the dimension we expect.
    return cost
def derivative_Cross_Entropy(y, labels):
    a = np.divide(labels,y)
    # print("a",a)
    b = np.divide(1-labels, 1-y)
    # print("b",b)
    dc = - (a - b)
    print("dc",dc)
    return dc





class Linear():
    def __init__(self,in_c,out_c):
        self.weight = np.random.randn(in_c, out_c) * 0.01
        print("self.weight",self.weight.shape)
        self.x = None
        self.y = None
    def update_weight(self,w):
        self.weight = w
    def get_weight(self):
        return self.weight
    def forward(self,x):
        self.x = x
        y = np.dot(x,self.weight)   # x = [21,2], W = [2,1] ,y = [21,1]
        self.y = y
        return y
    def backward(self,grad):
        #z2 = a1w2 --> dw2 = a1.T @ G
        grad_for_weight = self.x.T @ grad
        return grad_for_weight
    def get_input(self):
        return self.x
    def get_result(self):
        return self.y
class Sigmoid():
    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-self,))  
    def derivative_sigmoid(self,x):
        return x * (1.0-x)
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self,x):
        self.x = x
        self.y = self.sigmoid(x)
        return self.y
    def backward(self,grad):
        # y = sigmoid(x)
        new_grad = grad * self.derivative_sigmoid(y)
        return new_grad
    





def backward2(labels, y, result_dict, fun_list,lr):
    # forward path
    # x -w0> z0 -activate> a0 -w1> z1 -activate> a1 -w2> z2 -activate> a2 == y  <-- MSE LOSS --> labels
    # w0 (2, 5)
    # z0 (5, 5)
    # a0 (5, 5)
    # w1 (5, 5)
    # z1 (5, 5)
    # a1 (5, 5)
    # w2 (5, 1)
    # z2 (5, 1)
    # a2 (5, 1)
    # y == a2
    """calculate derivatives """
    dL_da2 =  derivative_Cross_Entropy(y,labels).mean()  # y 對 L 作微分
    da2_dz2 = derivative_sigmoid(result_dict['a2']) # a2 = sigomid(z2) - 微分 -> sigmoid(z2) * (1.0-sigmoid(z2))
    dz2_dw2 = result_dict['a1'] # z2 = a1w2
    
    dz2_da1 = result_dict['w2'].T # z2 = a1w2
    da1_dz1 = derivative_sigmoid(result_dict['a1']) # a1 = sigmoid(z1) 
    dz1_dw1 = result_dict['a0'] # z1 = a0w1
    dz1_da0 = result_dict['w1'].T # z1 = a0w1
    da0_dz0 = derivative_sigmoid(result_dict['a0']) #a0 = sigmoid(z0)
    dz0_dw0 = result_dict['x'] # z0 = xw0

    """calculate gradients """
    grad_dict = {}
    dL_dz2 = dL_da2 * da2_dz2
    assert dL_dz2.shape == result_dict['z2'].shape
    grad_dict['w2'] =dz2_dw2.T @ dL_dz2  
    assert grad_dict['w2'].shape == result_dict['w2'].shape

    
    dL_dz1 = dL_dz2 @ dz2_da1 * da1_dz1
    assert dL_dz1.shape == result_dict['z1'].shape
    grad_dict['w1'] = dz1_dw1.T @ dL_dz1 
    assert grad_dict['w1'].shape == result_dict['w1'].shape

    dL_dz0 = dL_dz1 @ dz1_da0 * da0_dz0
    assert dL_dz0.shape == result_dict['z0'].shape
    grad_dict['w0'] = dz0_dw0.T @ dL_dz0 
    assert grad_dict['w0'].shape == result_dict['w0'].shape

    """update weights """
    for idx in range(len(fun_list)):#['w0','w1','w2']
        weight_str = f"w{idx}"
        fun_list[idx].update_weight(result_dict[weight_str] - lr*grad_dict[weight_str])
def backward(labels, y, result_dict, fun_list,lr):
    m = y.shape[0]
    grad_dict = {}
     # x -w0> z0 -activate> a0 -w1> z1 -activate> a1 -w2> z2 -activate> a2 == y  <-- LOSS --> labels
    # y = a2
    dL_a2 = derivative_Cross_Entropy(y,labels) # L = loss(a)
    dL_z2 = dL_a2 * derivative_sigmoid(result_dict['a2']) # a0 = sigmoid(z0)
    grad_dict['w2'] = 1/m * np.dot(result_dict['a1'].T , dL_z2) # z2 = a1w2 --> dw2 = a1.T @ G


    dL_a1 =  dL_z2 @ result_dict['w2'].T # z2 = a1w2 --> da1 = G @ w2.T
    dL_z1 = np.array(dL_a1,copy=True) # a1 = relu(z1)
    dL_z1[result_dict['z1']<=0] = 0
    grad_dict['w1'] = 1/m * np.dot(result_dict['a0'].T , dL_z1) # z1 = a0w1 --> dw1 = a0.T @ G
    
    dL_a0 = dL_z1 @ result_dict['w1'].T # z1 = a0w1 --> da0 = G @ w1T
    dL_z0 = np.array(dL_a0,copy=True)  # a0 = relu(z0)
    dL_z0[result_dict['z0']<=0] = 0
    grad_dict['w0'] = 1/m * np.dot(result_dict['x'].T , dL_z0) # z0 = xw0 --> dw0 = x.T @ G

    
    """update weights """
    for idx in range(len(fun_list)):#['w0','w1','w2']
        
        weight_str = f"w{idx}"
        # print("grad_dict[weight_str]",grad_dict[weight_str])
        fun_list[idx].update_weight(result_dict[weight_str] - lr*grad_dict[weight_str])


def train(data, labels,fun_list,lr):
    
    result_dict = {'x':data}    # for record path
    """ forward """
    y = data.copy()
    for idx,linear_f in enumerate(fun_list):
        y = linear_f.forward(y)
        w = linear_f.get_weight()
        # print(f"w{idx}",w.shape)
        result_dict[f"w{idx}"] = w
        # print(f"z{idx}",y.shape)
        result_dict[f'z{idx}'] = y
        if idx == len(fun_list)-1:
            y = sigmoid(y)   
            print("last") 
        else:
            y = relu(y)
        # print(f"a{idx}",y.shape)
        result_dict[f'a{idx}'] = y
    
    # print(labels)
    
    
    loss = Cross_Entropy(y,labels)


    """backward"""
    backward( labels, y, result_dict, fun_list,lr)

    return loss,y

def show_results(x,y,pred_y):
    plt.subplot(1,3,1)
    plt.title("Loss",fontsize=18)
    plt.plot(loss_list)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.subplot(1,3,2)
    plt.title("Ground truth",fontsize=18)
    plot_data(x,y)
    plt.subplot(1,3,3) 
    plt.title("predict results",fontsize=18)
    plot_data(x,pred_y)
    plt.show()


if __name__ == '__main__':
    
    np.random.seed(5)
    data,labels = generate_linear(n=100)
    # plot_data(data,labels)
    # data,labels = generate_XOR_easy()    
    # plot_data(data,labels)
    """create model"""
    channels = [2,8,4,1]
    fun_list = [Linear(in_c,out_c)  for in_c,out_c in zip(channels[:-1],channels[1:])]
    """ start train"""
    steps = 500
    loss_list = []
    acc_list = []
    pred_y = None
    lr = 0.007
    for step in range(steps):
       
        loss,y = train(data, labels,fun_list,lr)
        
        loss_list.append(loss)
        pred_y = y
        print(f"[step:{step+1}/{steps}] loss: {loss}")

        if step % 100:
            lr = lr * 0.5

    # for a,b,c in zip(loss_list,pred_y,labels):
    #     print(a,b,c)
    # # print(loss_list)
    print(pred_y)
    
    
    show_results(data,labels,1.0*(pred_y>0.5))
