import numpy as np
import matplotlib.pyplot as plt
from data import generate_linear,generate_XOR_easy
from model import Linear,Sigmoid,Relu
from loss import cross_Entropy2,derivative_cross_Entropy2,cross_Entropy,derivative_cross_Entropy,mse,derivative_mse


def train(x, labels,fun_list,lr):
    """ forward """
    y = x.copy()
    for f in fun_list:
        y = f.forward(y)
    loss = mse(y,labels)
    """backward"""
    grad = derivative_mse(y,labels)
    fun_list.reverse()
    for f in fun_list:
        grad = f.backward(grad,lr)
    fun_list.reverse()

    return loss,y

def show_results(data,labels,pred_y):

    def _plot_data(data,labels):
        color_label = ['red' if i==0 else 'blue' for i in labels]
        plt.scatter(data[:,0],data[:,1],c= color_label)

    plt.subplot(1,2,1)
    plt.title("Ground truth",fontsize=18)
    _plot_data(data,labels)

    plt.subplot(1,2,2) 
    plt.title("predict results",fontsize=18)
    _plot_data(data,pred_y)
    plt.show()

if __name__ == '__main__':
    np.random.seed(5)
    data,labels = generate_linear(n=100)
    # data,labels = generate_XOR_easy() 
    channels = [2,6,6,1]
    fun_list = []
    for idx,(in_c,out_c) in enumerate(zip(channels[:-1],channels[1:])):
        fun_list.append(Linear(in_c,out_c))
        if idx == len(channels)-2: # last layer
            fun_list.append(Sigmoid())
        else:
            fun_list.append(Relu())
    steps = 50000
    lr = 0.0003
    loss_list = []
    pred_list = []
    for step in range(steps):
        loss,y = train(data, labels,fun_list,lr)
        loss_list.append(loss)
        pred_list.append(1.0*(y>0.5))
        # if step % 100 == 0:
        #     lr = lr* 0.09

    plt.plot(loss_list)
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
    # print(1.0*(pred_y>0.48))
    show_results(data,labels,pred_list[-1])


