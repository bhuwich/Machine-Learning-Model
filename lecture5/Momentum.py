import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':

    #set parameter
    lr = 0.003
    alpha = 0.9

    #preparedata for plot
    x_range = np.arange(-6, 5, 0.1)
    y_range = 2*(x_range**4) + 5*(x_range**3)- 30*(x_range**2) + x_range -1

    
    x = 4

    #iteration
    d_x = 0 
    while True:
        
        #cost function เมื่อเทียบกับ weight x
        y =  2*(x**4)+ 5*(x**3) - 30*(x**2) + x -1
        


        # คำนวณ gradient
        g_x = 8*(x**3) + 15*(x**2) - 60*x +1
        plt.figure(1)
        plt.clf()
        plt.plot(x_range, y_range, 'b-')
        plt.plot(x, y, 'ro')
        plt.grid()
        plt.xlabel("x")
        plt.ylabel("f(x)")       
        plt.show(block=False)
        plt.pause(0.001)
        time.sleep(0.5)
        

        #update weight
        d_x = lr*g_x + alpha*d_x
        x = x - d_x

