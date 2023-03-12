import matplotlib.pyplot as plt
import numpy as np

class CircleClassifier:
    def __init__(self):
        self.h = None
        self.k = None
        self.r = None
    
    def train(self, x, t, max_epocch=10000, lr=0.1, alpha=0.9):

        #convert to numpy array
        x =np.array(x)
        y =np.array(t)

        d_h = 0
        d_k = 0
        d_r = 0

        #initial weight 
        self.h = np.random.randn(1)[0]
        self.k = np.random.rand(1)[0]
        self.r = np.random.rand(1)[0]

        while True:

            # feed forward

            z = (x[:,0]-self.h)**2 + (x[:,1]-self.k)**2 - self.r**2
            y = self.tanh(z)
            n = len(t)
            #compute gradient
            g_h = (2/n)*np.sum((y-t)*(1-(self.tanh(z)**2))*2*(x[:,0]-self.h)*(-1))
            g_k = (2/n)*np.sum((y-t)*(1-(self.tanh(z)**2))*2*(x[:,1]-self.h)*(-1))
            g_r = (2/n)*np.sum((y-t)*(1-(self.tanh(z)**2))*(-2*self.r))


            #compute delta
            d_h = lr*g_h + alpha*d_h
            d_k = lr*g_k + alpha*d_k
            d_r = lr*g_r + alpha*d_r


            #plot
            fig =plt.figure(1)
            plt.clf()
            plt.plot(x[t==-1,0],x[t==-1,1], 'b*')
            plt.plot(x[t==1,0],x[t==1,1],'r*')
            
            theta = np.arange(0,2*np.pi+0.01,0.01)
            a = self.h + self.r*np.cos(theta)
            b = self.k + self.r*np.sin(theta)
            plt.plot(a,b, 'k-')
            plt.plot(self.h,self.k,'k*')
            plt.grid()
            plt.show(block=False)
            plt.pause(0.001)
            # update weights
            self.h = self.h - d_h
            self.k = self.k - d_k
            self.r = self.r - d_r
        
    def tanh(self, x):
        return (np.exp(x)- np.exp(-x))/(np.exp(x) +np.exp(-x))

if __name__ == '__main__':
    import pandas as pd
    data_filename = 'circle_data.csv'

    #load data
    df = pd.read_csv(data_filename)
    x = df.iloc[:,0].to_numpy()
    y = df.iloc[:,1].to_numpy()
    t = df.iloc[:,2].to_numpy()
    t[t==0] = -1
    x = np.transpose([x,y])

    network = CircleClassifier()
    network.train(x,t)