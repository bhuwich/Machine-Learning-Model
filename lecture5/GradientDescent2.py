import numpy as np
import matplotlib.pyplot as plt
import time
from PlotSurf3D import plotSurf
if __name__ == '__main__':
    
    #set parameter
    lr = 0.1
    
    #data set
    p = np.array([199000, 245000, 319000,240000, 312000, 279000, 310000, 405000, 405000, 324000])
    x = np.array([1100, 1400, 1425, 1550, 1600, 1700, 1700, 2350, 2350, 2450])
    N = len(p)

    
    #data preprocessing
    x = x * 0.001
    p = p * 0.00001

    #prepare data for plot 3d


    tmp = np.arange(-4,6.1,0.1)
    m_range = np.array([tmp for _ in range(len(tmp))])
    c_range = np.transpose(m_range)
    J_range = np.zeros(m_range.shape)
    for i in range(m_range.shape[0]):
        for j in range(c_range.shape[1]):
            y = m_range[i,j]*x + c_range[i,j]
            J_range[i,j] = (1/N)*np.sum((y-p)**2)
   

    #สุ่มค่า weight
    m = 4#np.random.rand(1)[0]
    c = 4#np.random.rand(1)[0]
    
    #iteration
    epoch = 0
    epoch_list = []
    J_list = []
    while True:
        y = m*x + c
        g_m = (2/N)*np.sum((y-p)*x)
        g_c = (2/N)*np.sum(y-p)
        #compute cost function
        J = (1/N)*np.sum((y-p)**2)
        print(f"epoch = {epoch}, cost = {J:.2f}")
        epoch_list.append(epoch)
        J_list.append(J)

        
        fig = plt.figure(1, figsize=(20,8))
        plt.clf()
        plt.subplot(2,2,1)
        plt.plot(x, p, 'b*')
        plt.plot(x, y, 'r-')
        plt.legend(['data',f'y= ({m:.2f})x + ({c:.2f})'])
        plt.grid()
        plt.xlabel("size *0.001")
        plt.ylabel('price * 0.00001')

        plt.subplot(2,2,3)
        plt.plot(epoch_list, J_list, 'r')
        plt.legend([f'cost = {J:.5f}'])
        plt.grid()
        plt.xlabel("epoch")
        plt.ylabel("cost")
        ax = fig.add_subplot(1,1,1, projection='3d')
        plotSurf(ax, m_range, c_range, J_range, curr_x =m, curr_y=c, curr_z =J, xlabel="m", ylabel="c", zlabel="J")
        plt.show(block=False)
        plt.pause(0.001)
        
        #update m,c
        #ลองปิดแต่ละตัวเพื่อเช็คว่าถูกมั้ย

        m = m - g_m*lr
        c = c- g_c*lr
        epoch+=1

    




