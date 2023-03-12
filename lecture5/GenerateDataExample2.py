import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

if __name__ == '__main__':

    N = 1000
    x_male = np.transpose([np.random.randn(N), np.random.randn(N)]) # feature 1 เช่น ส่วนสูง
    t_male = -1*np.ones(x_male.shape[0])
    x_female = np.transpose([np.random.randn(N), np.random.randn(N)]) +2 # feature 1 เช่น ส่วนสูง
    t_female = 1*np.ones(x_female.shape[0])
    x=np.append(x_male,x_female, axis =0)
    t = np.append(t_male,t_female, axis =0)


    print(x.shape)
    print(t.shape)

    sio.savemat("data.mat",{'x': x, 't': t})
    plt.figure(1)
    plt.plot(x[t==-1,0],x[t==-1,1],'b.')
    plt.plot(x[t==1,0],x[t==1,1],'r.')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(['t = -1', 't = 1'])
    plt.grid()
    plt.show()