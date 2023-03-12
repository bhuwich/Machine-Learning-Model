import numpy as np
import time
class Perceptron:
    def __init__(self):
        self.lr = 0.01
        self.w = None
        self.b = None
        self.c1 = "1"
        self.c2 = "2"

    def predict(self, x):
        #x = [[1.2,3.5],[2.5,1.4]]
        output = []
        for i in range(len(x)):
            output.append(self._predict(x[i]))
        return output
    
    def _predict(self,x_):
        net = self.dotProduct(x_)
        #x_ = [1.2,3.5]
        #w = [0.1,0.3]
        #b =2.3
        if net > 0:
            return self.c1
        else:
            return self.c2
    
    def dotProduct(self,x_):
        net = 0
        for i in range(len(x_)):
            net += x_[i] * self.w[i]
        net += self.b
        return net
    
    def train(self,x, t):
        #convert to numpy
        x = np.array(x)
        t = np.array(t)

        #random weight
        self.w = np.random.randn(x.shape[1])
        self.b = np.random.randn(1)[0]
        #iterative training
        epoch = 0
        old_J = np.inf
        while True:
        
            J = self.cost(x,t)
            print(f"epoch = {epoch}, cost = {J:.2f}")
            if J >= old_J:
                break
            old_J = J
            time.sleep(1)
            epoch += 1
            W = np.append(self.w,self.b)
            g_vec = np.zeros(x.shape[1]+ 1)
            for i in range(len(x)):
                if self._predict(x[i]) != t[i]: #error
                    x_tmp = np.append(x[i],1)
                    if t[i] == self.c1:
                        g_vec += -x_tmp
                    else:
                        g_vec += x_tmp
            #update weight W = [W[0],W[1],b]
            #g_vec = delta*[x_[0],x_[1],1] ->[x_[0],x_[1],1] ,มาจากเซตของ error
            #W = W - lr*g_vec
            W = W- self.lr*g_vec

            self.w = W[:-1]
            self.b = W[-1]

    def cost(self,x ,t):
        J = 0
        for i in range(len(t)):
            x_ = x[i]
            net_ = self.dotProduct(x_)
            t_ = t[i]
            if net_ <= 0 and t_ == self.c1:
                J += np.abs(net_)
            elif net_ > 0 and t_ == self.c2:
                J += np.abs(net_)
        return J
if __name__ == "__main__":
    x = [[1,2],[2.5,3],[3,1],[-3,-1.7],[-1.6,-3],[1.5,-3]]
    t = ['1','1','1','2','2','2']

    #create model
    perceptron = Perceptron()
    perceptron.train(x,t)
    from PlotHyperPlane2d import plotHyperPlane2d
    import matplotlib.pyplot as plt
    plotHyperPlane2d(x,t, perceptron.w,perceptron.b)
    plt.grid()
    plt.show()