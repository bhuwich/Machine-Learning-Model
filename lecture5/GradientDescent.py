import numpy as np
import matplotlib.pyplot as plt
import time
if __name__ == '__main__':
    
    #set parameter
    lr = 0.8
    
    #data set
    p = np.array([199000, 245000, 319000,240000, 312000, 279000, 310000, 405000, 405000, 324000])
    x = np.array([1100, 1400, 1425, 1550, 1600, 1700, 1700, 2350, 2350, 2450])
    N = len(p)

    
    #data preprocessing
    x = x * 0.001
    p = p * 0.00001

    #จัดเตรียมข้อมูลสำหรับการplot
    range_m = np.arange(-200, 200, 0.1)
    range_J = np.zeros(len(range_m))
    for i in range(len(range_m)):
        y = range_m[i]*x
        J = (1/N)*np.sum((y-p)**2)
        range_J[i] = J

    #สุ่มค่า m ขึ้นมา
    m = np.random.rand(1)[0]

    #iterative
    epoch =0
    J_list = []
    epoch_list = []
    while True:

        #prediction
        y = m*x

        #gradient of m, gJ(m)
        g_m =(2/N)*np.sum((y-p)*x)

        #compute cost function
        J = (1/N)*np.sum((y-p)**2)
        print(f"epoch {epoch}, J = {J:.2f}, g_m = {g_m:.2f}")
        J_list.append(J)
        epoch_list.append(epoch)
        time.sleep(1)
        
        # plot
        plt.figure(1, figsize=(20,8))
        plt.clf()
        plt.subplot(2,2,1)
        plt.plot(x, p, 'b*')
        plt.plot(x, y, 'r-')
        plt.legend(['data',f'y= ({m:.2f})x'])
        plt.grid()
        plt.xlabel("size *0.001")
        plt.ylabel('price * 0.00001')

        plt.subplot(2,2,3)
        plt.plot(epoch_list, J_list, 'r')
        plt.legend([f'cost = {J:.5f}'])
        plt.grid()
        plt.xlabel("epoch")
        plt.ylabel("cost")

        plt.subplot(1,2,2)
        plt.plot(range_m, range_J, 'b-')
        plt.plot(m, J, 'ro')
        plt.text(m, J, f"g_m = {g_m:.5f}\ncost = {J:.5f}")
        plt.grid()
        plt.xlabel('m')
        plt.ylabel('cost')    

        plt.show(block=False)
        plt.pause(0.001)

        #update m
        m = m -lr*g_m
        epoch +=1