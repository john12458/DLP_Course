import numpy as np
import matplotlib.pyplot as plt
from data import generate_linear,generate_XOR_easy
from model import Linear,Sigmoid,Relu
from loss import MSE,Cross_Entropy
import argparse

def train(x, labels,fun_list,lr,loss_f):
    """ forward """
    y = x.copy()
    for f in fun_list:
        y = f.forward(y)
    loss = loss_f.forward(y,labels)
    """backward"""
    grad = loss_f.backward()
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

def main(args):
    if args.wandb:
        import wandb
        wandb.init(project="dlp", entity="kycj")
        unit_str = "" 
        for unit in args.unit:
            unit_str += str(unit) + "_"
        act_str = "activate" if args.use_activate else ""
        wandb.run.name = f"{args.data}_{unit_str}_{act_str}"
        wandb.config.update(args)
        
    np.random.seed(5)
    lr,steps = args.lr,args.step
    """ prepare data"""
    data=None
    labels=None
    if args.data == "xor":
        data,labels = generate_XOR_easy() 
    else:
        data,labels = generate_linear(n=100)
    """ prepare loss """
    loss_f = None
    if args.loss == "mse":
        loss_f =  MSE()
    else:
        loss_f =  Cross_Entropy()
    print("loss: ",loss_f.__class__.__name__)

    channels = [2] + args.unit + [1]
    """ create model """
    print("channel: ",channels)
    print("use activate") if args.use_activate else print("without activate")
    fun_list = []
    for idx,(in_c,out_c) in enumerate(zip(channels[:-1],channels[1:])):
        fun_list.append(Linear(in_c,out_c))
        if args.use_activate:
            activate_f = Sigmoid() if idx == len(channels[:-1])-1 else Relu() # last layer use sigmoid
            fun_list.append(activate_f)
    """ start train """
    loss_list = []
    acc_list = []
    lr_list = []
    pred_y = None
    for step in range(steps):
        loss,y = train(data, labels,fun_list,lr,loss_f)
        loss_list.append(loss)
        pred_y = 1.0*(y>0.5)
        acc = (labels== pred_y).mean()
        acc_list.append(acc)
        lr_list.append(lr)
        if step % 10 ==0:
            print(f"[step:{step+1}/{steps}] loss: {loss:4f} | acc:{acc} | lr:{lr:8f}")
        if args.wandb:
            wandb.log({"loss":loss,"acc":acc,"lr":lr})
        if step % (steps//10) == 0:
            lr = lr* 0.9
        
    """ show result """
    if not args.wandb:
        # loss
        plt.subplot(2,1,1)
        plt.plot(loss_list)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        # acc
        plt.subplot(2,1,2)
        plt.plot(acc_list)
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.show()
        # lr
        plt.figure()
        plt.plot(lr_list)
        plt.ylabel('lr')
        plt.xlabel('epoch')
        plt.show()
  
    # result
    show_results(data,labels,pred_y)
    if not args.wandb:
        plt.show() 

    if args.wandb:
        wandb.log({"result": wandb.Image(plt)})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default= "linear",type=str,help="linear,xor")
    parser.add_argument("--loss", default= "cross_entropy",type=str,help="mse,cross_entropy")
    parser.add_argument("--lr", default=0.0003 ,type=float,help="learning rate")
    parser.add_argument("--step",default=50000,type=int,help="iteration you train")
    parser.add_argument('--unit',default=[6,6], type=int, nargs='+')
    parser.add_argument('--use_activate',default=False, action="store_true")
    parser.add_argument('--wandb',default=False, action="store_true")
    

    args = parser.parse_args()

    main(args)


