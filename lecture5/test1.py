import matplotlib.pyplot as plt
import numpy as np

class ParabolicClassifier:
    def __init__(self):
        self.a = None
        self.b = None
        self.c = None
    
    def _predict(self, x_):
        return self.tanh(self.a*x_[0]**2 + self.b*x_[0] + self.c - x_[1])
    
    def predict(self, x):
        return np.array([self._predict(x[i]) for i in range(len(x))])
    
    def tanh(self, x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    
    def train(self, x, t, lr=0.01, alpha=0.9):

        x = np.array(x)
        t = np.array(t)

        self.a = np.random.randn(1)[0]
        self.b = np.random.randn(1)[0]
        self.c = np.random.randn(1)[0]

        d_a = 0
        d_b = 0
        d_c = 0
        J_list = []
        epoch_list = []
        epoch = 0
        while True:
            z = []
            y = []
            for i in range(len(x)):
                x_ = x[i]
                z_ = self.a*x_[0]**2 + self.b*x_[0] + self.c - x_[1]
                y_ = self.tanh(z_)
                z.append(z_)
                y.append(y_)
            z = np.array(z)
            y = np.array(y)

            J = self.cost(y, t)
            print(J)
            J_list.append(J)
            epoch_list.append(epoch)
            epoch += 1

            x_tmp = np.arange(min(x[:,0]), max(x[:,0]), 0.01)
            y_tmp = self.a*x_tmp**2 + self.b*x_tmp + self.c
            plt.figure(1)
            plt.clf()
            plt.plot(x[t==-1, 0], x[t==-1,1], 'b*')
            plt.plot(x[t==1,0], x[t==1,1], 'r*')
            plt.plot(x_tmp, y_tmp, 'k')
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.legend(["class 0", "class 1", "parabola"])
            plt.grid()
            plt.ylim([-5,10])
            plt.show(block=False)
            plt.pause(0.001)


            N = len(t)
            g_a = (1/N)*np.sum(2*(y-t)*(1-self.tanh(z)**2)*(x[:,0])**2)
            g_b = (1/N)*np.sum(2*(y-t)*(1-self.tanh(z)**2)*x[:,0])
            g_c = (1/N)*np.sum(2*(y-t)*(1-self.tanh(z)**2))

            d_a = lr*g_a + alpha*d_a
            d_b = lr*g_b + alpha*d_b
            d_c = lr*g_c + alpha*d_c

            self.a = self.a - d_a
            self.b = self.b - d_b
            self.c = self.c - d_c

    def cost(self, y, t):
        return (1/len(t))*np.sum((y-t)**2)

if __name__ == '__main__':
    n = 1000

    x_1_all = np.random.randn(n)
    x_2_all = np.random.randn(n)

    t = []
    for i in range(len(x_1_all)):
        if x_1_all[i] < x_2_all[i]**2 + np.random.randn(1)[0]*0.1:
            t.append(1)
        else:
            t.append(-1)
    t = np.array(t)
    x = np.transpose([x_2_all,x_1_all])

    parabolic_classifier = ParabolicClassifier()
    parabolic_classifier.train(x, t,lr=0.1, alpha=0.9)