import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from matplotlib import cm
import pandas as pd

class NeuralNetwork:
    def __init__(self):
        self.w1_11 = None
        self.w1_12 = None
        self.b1_1 = None
        self.w1_21 = None
        self.w1_22 = None
        self.b1_2 = None
        self.w1_31 = None
        self.w1_32 = None
        self.b1_3 = None
        self.w1_41 = None
        self.w1_42 = None
        self.b1_4 = None
        self.w2_11 = None
        self.w2_12 = None
        self.w2_13 = None
        self.w2_14 = None
        self.b2_1 = None
        
    def _predict(self, x_):
        z1_1_ = self.w1_11*x_[0] + self.w1_12*x_[1] + self.b1_1
        z1_2_ = self.w1_21*x_[0] + self.w1_22*x_[1] + self.b1_2
        z1_3_ = self.w1_31*x_[0] + self.w1_32*x_[1] + self.b1_3
        #z1_4_ = self.w1_41*x_[0] + self.w1_42*x_[1] + self.b1_4
        o1_1_ = self.tanh(z1_1_)
        o1_2_ = self.tanh(z1_2_)
        o1_3_ = self.tanh(z1_3_)
        #o1_4_ = self.tanh(z1_4_)
        z2_1_ = self.w2_11*o1_1_ + self.w2_12*o1_2_ + self.w2_13*o1_3_ + self.b2_1
        o2_1_ = self.tanh(z2_1_)
        return o2_1_
    
    def tanh(self, x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x) + np.exp(-x))
    
    def predict(self, x):
        return np.array([self._predict(x[i]) for i in range(len(x))])
    
    def cost(self, y, t):    
        return (1/len(t))*np.sum((y-t)**2) #Mean square error
    
    def train(self, x,t, lr=0.01, alpha =0.9):
        #convert to numpy array
        x = np.array(x)
        t = np.array(t)


        x_plot = []
        range_i = np.arange(min(x[:,0]), max(x[:,0]), 0.04)
        range_j = np.arange(min(x[:,1]), max(x[:,1]), 0.04)
        for i in range_i:
            for j in range_j:
                x_plot.append([i,j])
        x_plot = np.array(x_plot)
        X = np.resize(x_plot[:,0], [len(range_i), len(range_j)])
        Y = np.resize(x_plot[:,1], [len(range_i), len(range_j)])
        
        
        #initialization weight
        self.w1_11 = np.random.rand(1)[0]
        self.w1_12 = np.random.rand(1)[0]
        self.b1_1 = np.random.rand(1)[0]
        self.w1_21 = np.random.rand(1)[0]
        self.w1_22 = np.random.rand(1)[0]
        self.b1_2 = np.random.rand(1)[0]
        self.w1_31 = np.random.rand(1)[0]
        self.w1_32 = np.random.rand(1)[0]
        self.b1_3 = np.random.randn(1)[0]
        self.w1_41 = np.random.randn(1)[0]
        self.w1_42 = np.random.randn(1)[0]
        self.b1_4 = np.random.randn(1)[0]
        self.w2_11 = np.random.rand(1)[0]
        self.w2_12 = np.random.rand(1)[0]
        self.w2_13 = np.random.rand(1)[0]
        self.w2_14 = np.random.rand(1)[0]
        self.b2_1 = np.random.rand(1)[0]
        d_w1_11 = 0
        d_w1_12 = 0
        d_b1_1 = 0
        d_w1_21 = 0
        d_w1_22 = 0
        d_b1_2 = 0
        d_w1_31 = 0
        d_w1_32 = 0
        d_b1_3 = 0
        d_w1_41 = 0
        d_w1_42 = 0
        d_b1_4 = 0
        d_w2_11= 0 
        d_w2_12 =0
        d_w2_13 = 0
        d_w2_14 = 0
        d_b2_1 = 0
        
        cost_list = []
        epoch_list = []
        epoch = 0
        
        while True:
            z1_1 = np.array([])
            z1_2 = np.array([])
            z1_3 = np.array([])
            z1_4 = np.array([])
            o1_1 = np.array([])
            o1_2 = np.array([])
            o1_3 =  np.array([])
            o1_4 = np.array([])
            z2_1 = np.array([])
            o2_1 = np.array([])
            for i in range(len(t)):
                
                
                x_ = x[i]
                z1_1_ = self.w1_11*x_[0] + self.w1_12*x_[1] + self.b1_1
                z1_2_ = self.w1_21*x_[0] + self.w1_22*x_[1] + self.b1_2
                z1_3_ = self.w1_31*x_[0] + self.w1_32*x_[1] + self.b1_3
                #z1_4_ = self.w1_41*x_[0] + self.w1_42*x_[1] + self.b1_4
                o1_1_ = self.tanh(z1_1_)
                o1_2_ = self.tanh(z1_2_)
                o1_3_ = self.tanh(z1_3_)
                #o1_4_ = self.tanh(z1_4_)
                z2_1_ = self.w2_11*o1_1_ + self.w2_12*o1_2_ + self.w2_13*o1_3_ + self.b2_1
                o2_1_ = self.tanh(z2_1_)
                z1_1 = np.append(z1_1,z1_1_)
                z1_2= np.append(z1_2,z1_2_)
                z1_3 = np.append(z1_3,z1_3_)
                #z1_4 = np.append(z1_4,z1_4_)
                o1_1= np.append(o1_1,o1_1_)
                o1_2= np.append(o1_2,o1_2_)
                o1_3 = np.append(o1_3,o1_3_)
                #o1_4 = np.append(o1_4,o1_4_)
                z2_1 = np.append(z2_1,z2_1_)
                o2_1 = np.append(o2_1,o2_1_)
                
            #gradient
            N = len(t)
            
            

            g_w1_11 = (2/N)*np.sum((o2_1-t)*(1-(self.tanh(z2_1)**2))*self.w2_11*(1-(self.tanh(z1_1)**2))*x[:,0])
            g_w1_12 = (2/N)*np.sum((o2_1-t)*(1-(self.tanh(z2_1)**2))*self.w2_11*(1-(self.tanh(z1_1)**2))*x[:,1])
            g_w1_21 = (2/N)*np.sum((o2_1-t)*(1-(self.tanh(z2_1)**2))*self.w2_12*(1-(self.tanh(z1_2)**2))*x[:,0])
            g_w1_22 = (2/N)*np.sum((o2_1-t)*(1-(self.tanh(z2_1)**2))*self.w2_12*(1-(self.tanh(z1_2)**2))*x[:,1])
            g_w1_31 = (2/N)*np.sum((o2_1-t)*(1-(self.tanh(z2_1)**2))*self.w2_13*(1-(self.tanh(z1_3)**2))*x[:,0])
            g_w1_32 = (2/N)*np.sum((o2_1-t)*(1-(self.tanh(z2_1)**2))*self.w2_13*(1-(self.tanh(z1_3)**2))*x[:,1])
            #g_w1_41 = (2/N)*np.sum((o2_1-t)*(1-(self.tanh(z2_1)**2))*self.w2_14*(1-(self.tanh(z1_4)**2))*x[:,0])
            #g_w1_42 = (2/N)*np.sum((o2_1-t)*(1-(self.tanh(z2_1)**2))*self.w2_14*(1-(self.tanh(z1_4)**2))*x[:,1])
            g_b1_1 = (2/N)*np.sum((o2_1-t)*(1-(self.tanh(z2_1)**2))*self.w2_11*(1-(self.tanh(z1_1)**2)))
            g_b1_2 = (2/N)*np.sum((o2_1-t)*(1-(self.tanh(z2_1)**2))*self.w2_12*(1-(self.tanh(z1_2)**2)))
            g_b1_3 = (2/N)*np.sum((o2_1-t)*(1-(self.tanh(z2_1)**2))*self.w2_13*(1-(self.tanh(z1_3)**2)))
            #g_b1_4 = (2/N)*np.sum((o2_1-t)*(1-(self.tanh(z2_1)**2))*self.w2_14*(1-(self.tanh(z1_4)**2)))
            g_b2_1 = (2/N)*np.sum((o2_1-t)*(1-(self.tanh(z2_1)**2)))
            g_w2_11 = (2/N)*np.sum((o2_1-t)*(1-(self.tanh(z2_1)**2))*o1_1)
            g_w2_12 = (2/N)*np.sum((o2_1-t)*(1-(self.tanh(z2_1)**2))*o1_2)
            g_w2_13 =(2/N)*np.sum((o2_1-t)*(1-(self.tanh(z2_1)**2))*o1_3)
            #g_w2_14 = (2/N)*np.sum((o2_1-t)*(1-(self.tanh(z2_1)**2))*o1_4)
            
            #compute cost
            cost = self.cost(o2_1,t)
            cost_list.append(cost)
            epoch_list.append(epoch)
            epoch +=1
            if(epoch >=150):
                plt.show()
                break
            print(f"epoch = {epoch}, cost = {cost}")
            
            
            
            #compute momentum
            d_w1_11 = lr*g_w1_11 + alpha*d_w1_11
            d_w1_12 = lr*g_w1_12 + alpha*d_w1_12
            d_b1_1 = lr*g_b1_1 + alpha*d_b1_1                
            d_w1_21 = lr*g_w1_21 + alpha*d_w1_21
            d_w1_22 = lr*g_w1_22 + alpha*d_w1_22
            d_b1_2 = lr*g_b1_2 + alpha*d_b1_2
            d_w1_31 = lr*g_w1_31 + alpha*d_w1_31
            d_w1_32 = lr*g_w1_32 + alpha*d_w1_32
            d_b1_3 = lr*g_b1_3 + alpha*d_b1_3
            #d_w1_41 = lr*g_w1_41 + alpha*d_w1_41
            #d_w1_42 = lr*g_w1_42 + alpha*d_w1_42
            #d_b1_4 = lr*g_b1_4 + alpha*d_b1_4
            d_w2_11= lr*g_w2_11 + alpha*d_w2_11
            d_w2_12 =lr*g_w2_12 + alpha*d_w2_12
            d_w2_13 = lr*g_w2_13 +alpha*d_w2_13
            #d_w2_14 = lr*g_w2_14 + alpha*d_w2_14
            d_b2_1 = lr*g_b2_1 + alpha*d_b2_1

            #linear model
            x_tmp = np.array([min(x[:,0]),max(x[:,1])])
            y = ((-self.w1_11/self.w1_12)*x_tmp) - (self.b1_1/self.w1_12)
            y2 = ((-self.w1_21/self.w1_22)*x_tmp) - (self.b1_2/self.w1_22)
            y3 = ((-self.w1_31/self.w1_32)*x_tmp) - (self.b1_3/self.w1_32)
            #y4 = ((-self.w1_41/self.w1_42)*x_tmp) - (self.b1_4/self.w1_42)
            x3_tmp = np.array([-1,1])
            output_y = ((-self.w2_11/self.w2_12)*x3_tmp)- (self.b2_1/self.w2_12)

            #plot
            fig = plt.figure(1,figsize=[20,8])
            plt.clf()
            plt.subplot(2,3,1)
            plt.plot(x[t==-1,0],x[t==-1,1],'r.')
            plt.plot(x[t==1,0],x[t==1,1],'b.')
            plt.plot(x_tmp,y,'g-')
            plt.plot(x_tmp,y2,'m-')
            plt.plot(x_tmp,y3,'k-')
            #plt.plot(x_tmp,y4,'y-')
            plt.xlabel("feature0")
            plt.ylabel("feature1")
            plt.legend(['class 0','class 1','linear 0','linear 1','linear 2'])
            plt.xlim([-2,2])
            plt.ylim([-2,2])
            plt.grid()
            ax = plt.subplot(1,3,2,projection='3d')
            ax.plot3D(o1_1[t==-1],o1_2[t==-1],o1_3[t==-1],'r.')
            ax.plot3D(o1_1[t==1],o1_2[t==1],o1_3[t==1],'b.')
            ax.set_xlabel("o1_1")
            ax.set_ylabel("o1_2")
            ax.set_zlabel("o1_3")
            #plt.plot(x3_tmp,output_y,'k-')
            # plt.xlim([-1.5,1.5])
            # plt.ylim([-1.5,1.5])
            # plt.zlim([-1.5,1.5])
            # plt.xlabel("o1_1")
            # plt.ylabel("o1_2")
            # plt.zlabel("o1_3")
            plt.grid()
            plt.subplot(2,3,4)
            plt.plot(epoch_list,cost_list,'r-')
            plt.xlabel('epoch_list')
            plt.ylabel('cost_list')
            plt.legend(['epoch,cost'])            
            plt.grid()

            ax1 = fig.add_subplot(1,3,3,projection='3d')
            Z = np.resize(self.predict(x_plot), [len(range_i), len(range_j)])
            ax1.plot_surface(X, Y, Z, cmap=cm.cool, alpha=0.5, linewidth=0, antialiased=False)
            ax1.plot3D(x[t==-1,0],x[t==-1,1],o2_1[t==-1], 'r*')
            ax1.plot3D(x[t==1,0],x[t==1,1],o2_1[t==1], 'b*')
            ax1.set_xlabel("x2")
            ax1.set_ylabel("x1")
            ax1.set_zlabel("o2_1")
            ax1.view_init(30, 60)

            plt.show(block=False)
            plt.pause(0.01)
            #plt.savefig(f'image/{epoch}.png')

            #update weight
            self.w1_11 = self.w1_11 - d_w1_11
            self.w1_12 = self.w1_12 - d_w1_12
            self.b1_1 = self.b1_1 - d_b1_1
            self.w1_21 = self.w1_21 - d_w1_21
            self.w1_22 = self.w1_22 - d_w1_22
            self.b1_2 = self.b1_2 - d_b1_2
            self.w1_31 = self.w1_31 -d_w1_31
            self.w1_32 = self.w1_32 - d_w1_32
            self.b1_3 = self.b1_3 - d_b1_3
            #self.w1_41 = self.w1_41 -d_w1_41
            #self.w1_42 = self.w1_42 -d_w1_42
            #self.b1_4 = self.b1_4 - d_b1_4
            self.w2_11 = self.w2_11 - d_w2_11
            self.w2_12 = self.w2_12 - d_w2_12
            self.w2_13 = self.w2_13 - d_w2_13
            #self.w2_14 = self.w2_14 - d_w2_14
            self.b2_1 = self.b2_1 - d_b2_1
if __name__ == '__main__':

    #set parameter
    data_filename = "data.mat"
    lr = 0.01
    alpha = 0.9

    #load data
    # n = 1000

    # x_1_all = np.random.randn(n)
    # x_2_all = np.random.randn(n)

    # t = []
    # for i in range(len(x_1_all)):
    #     if x_1_all[i] < x_2_all[i]**2 + np.random.randn(1)[0]*0.1:
    #         t.append(1)
    #     else:
    #         t.append(-1)
    # t = np.array(t)
    # x = np.transpose([x_2_all,x_1_all])
    # data = sio.loadmat(data_filename)
    # x = data['x']
    # t = data['t'][0]
    df = pd.read_csv("circle_data.csv")
    x = df.iloc[:,0].to_numpy()
    y = df.iloc[:,1].to_numpy()
    t = df.iloc[:,2].to_numpy()
    t[t==0] = -1
    x = np.transpose([x,y])



    #create object
    nn = NeuralNetwork()
    nn.train(x,t,lr=0.1,alpha=0.9)
                                                                                    
            
                
        