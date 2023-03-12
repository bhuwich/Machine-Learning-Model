import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

class NeuralNetwork:
    def __init__(self):
        self.w1_11 = None
        self.w1_12 = None
        self.b1_1  = None
    
    def tanh(self, x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x) + np.exp(-x))

    def _predict(self, x_):
        z1_1_ = x_[0]*self.w1_11 + x_[1]*self.w1_12 + self.b1_1
        o1_1_ = self.tanh(z1_1_)
        return o1_1_
    
    # def predict(self,x):
    #     output = []
    #     for i in range(len(x)):
    #         output.append(self._predict(x[i]))
    #     return np.array(output)

    def predict(self,x):
        return np.array([self._predict(x[i]) for i in range(len(x))])
    
    def cost(self, y, t):    
        return (1/len(t))*np.sum((y-t)**2) #Mean square error

    def train(self, x, t, lr=0.01, alpha=0.9):

        #convert to numpy array
        x = np.array(x)
        t = np.array(t)

        # prepare plot
        plot_range = np.arange(-4,5,0.1)
        x_plot = []
        for i in range(len(plot_range)):
            for j in range(len(plot_range)):
                x_plot.append([plot_range[i], plot_range[j]])
        x_plot = np.array(x_plot)
        #x_plot = [[-4, -3.9],
        #          [-4, -3.8],
        #           ...
        #            [-3.9, -4],
        #            [-3.9, -3.9],
        #            ....
        #            [4,-4],
        #             ....
        # ]


        #initialize weight
        self.w1_11 = np.random.randn(1)[0]
        self.w1_12 = np.random.randn(1)[0]
        self.b1_1 = np.random.randn(1)[0]

        d_w1_11 = 0
        d_w1_12 = 0
        d_b1_1 = 0
        cost_list = []
        epoch_list = []
        epoch = 0
        while True:

            # feed forward
            o1_1 = []
            z1_1 = []
            for i in range(len(t)):
                x_ = x[i]
                z1_1_ = self.w1_11*x_[0] + self.w1_12*x_[1] + self.b1_1 #dot product
                o1_1_ = self.tanh(z1_1_)
                z1_1.append(z1_1_)
                o1_1.append(o1_1_)
            z1_1 = np.array(z1_1)
            o1_1 = np.array(o1_1)
            print(x[:,0])
            print(x[:,1])
            #compute gradient
            g_w1_11 = (2/len(t))*np.sum((o1_1-t)*(1-(self.tanh(z1_1)**2))*x[:,0])
            g_w1_12 = (2/len(t))*np.sum((o1_1-t)*(1-(self.tanh(z1_1)**2))*x[:,1])
            g_b1_1 = (2/len(t))*np.sum((o1_1-t)*(1-(self.tanh(z1_1)**2)))

            #visualization
            #compute cost
            cost = self.cost(o1_1,t)
            cost_list.append(cost)
            epoch_list.append(epoch)
            epoch +=1
            #linear model
            m_lm = -self.w1_11/self.w1_12
            c_lm = -self.b1_1/self.w1_12
            x_lm = np.array([min(x_plot[:,0]), max(x_plot[:,0])])
            y_lm = m_lm*x_lm + c_lm

            #prepare y_plot
            y_plot = self.predict(x_plot)
            plt.figure(1, figsize=(10,5))
            plt.clf()
            plt.subplot(1,2,1)
            plt.plot(x_plot[y_plot<0,0], x_plot[y_plot<0,1], 'c.')
            plt.plot(x_plot[y_plot>=0,0], x_plot[y_plot>=0,1], 'm.')    
            plt.plot(x[t==-1,0],x[t==-1,1],'b.')
            plt.plot(x[t==1,0],x[t==1,1],'r.')
            plt.plot(x_lm, y_lm, 'k-')
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.legend(['class 0','class 1','target 0', 'target 1','linear model'])
            plt.xlim([-4,5])
            plt.ylim([-4,5])
            plt.grid()


            plt.subplot(1,2,2)
            plt.plot(epoch_list,cost_list,'r-')
            plt.legend([
                f"cost ={cost:.5f}\n" + \
                f"g_w1_11 = {g_w1_11:.5f}\n" + \
                f"g_w1_12 = {g_w1_12:.5f}\n" + \
                f"g_b1_1 = {g_b1_1:.5f}\n" + \
                f"d_w1_11 = {d_w1_11:.5f}\n" + \
                f"d_w1_12 = {d_w1_12:.5f}\n" + \
                f"d_b1_1 = {d_b1_1:.5f}"        
            ])
            plt.grid()
            plt.show(block=False)
            plt.pause(0.001)

            #compute momentum
            d_w1_11 = lr*g_w1_11 + alpha*d_w1_11
            d_w1_12 = lr*g_w1_12 + alpha*d_w1_12
            d_b1_1 = lr*g_b1_1 + alpha*d_b1_1

            #update weight
            self.w1_11 = self.w1_11 - d_w1_11
            self.w1_12 = self.w1_12 - d_w1_12
            self.b1_1 = self.b1_1 - d_b1_1

        


if __name__ == '__main__':

    #set parameter
    data_filename = "data.mat"
    lr = 0.01
    alpha = 0.9

    #load data
    data = sio.loadmat(data_filename)
    x = data['x']
    t = data['t'][0]

    #create object
    nn = NeuralNetwork()
    nn.train(x,t,0.1)