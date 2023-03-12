import numpy as np
import matplotlib.pyplot as plt
import time
from PlotSurf3D import plotSurf

if __name__ == '__main__':

    #set parameter
    lr = 0.003
    alpha = 0.6

    #preparedata for plot
    x_range_list = []
    x_range = np.arange(-6, 5, 0.1)
    
    
    for i in range(len(x_range)):
        x_range_list.append(x_range)
    x_range_list = np.array(x_range_list)
    y_range = np.transpose(x_range_list)
    fxy_range = np.zeros(x_range_list.shape)
    for i in range(x_range_list.shape[0]):
        for j in range(x_range_list.shape[1]):
            fxy_range[i,j] = 2*(x_range_list[i,j]**4) + 2*(y_range[i,j]**4) + 5*(x_range_list[i,j]**3) + 5*(y_range[i,j]**3) - 30*(x_range_list[i,j]**2) - 30*(y_range[i,j]**2) +x_range_list[i,j] +y_range[i,j] -1

    
    


    
    x = 4
    y = 4

    #iteration
    d_x = 0
    d_y =0
    epoch =0
    epoch_list = []
    fxy_list = []
    while True:
        
        #cost function เมื่อเทียบกับ weight x
        fxy = 2*(x**4) + 2*(y**4) + 5*(x**3) + 5*(y**3) - 30*(x**2) - 30*(y**2) +x +y -1
        fxy_list.append(fxy)
        epoch_list.append(epoch)
        print(f"f(x) = {fxy:.2f}, epoch = {epoch}")
        print(x_range_list.shape,y_range.shape,fxy_range.shape)
        #plot
        fig =plt.figure(1, figsize=(20,8))
        plt.clf()
        plt.subplot(1,2,1)
        plt.plot(epoch_list,fxy_list, 'r')
        plt.xlabel("epoch")
        plt.ylabel("f(x,y)")  
        plt.grid()      
        ax = fig.add_subplot(1,2,2, projection='3d')
        plotSurf(ax, x_range_list, y_range, fxy_range, curr_x =x, curr_y=y, curr_z =fxy, xlabel="x", ylabel="y", zlabel="fxy")
        plt.show(block=False)
        plt.pause(0.001)
        # คำนวณ gradient
        g_x = 8*(x**3) + 15*(x**2) - 60*x +1
        g_y = 8*(y**3) + 15*(y**2) - 60*y +1
        
        

        #update weight x
        d_x = lr*g_x + alpha*d_x
        x = x - d_x
        #update weight y
        d_y = lr*g_y + alpha*d_y
        y = y - d_y
        epoch+=1