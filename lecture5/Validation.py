from re import A
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    lr = 0.002 
    # ชุดข้อมูล Train
    x_train = np.array([-1,1,8])
    t_train = np.array([1,5,6])

    # ชุดข้อมูล Validation
    x_val = np.array([0,2,4,5,8])
    t_val = np.array([2,3,5,3,7])

    # ชุดข้อมูล Testing
    x_test = np.array([-2,0.5,2,6,10])
    t_test = np.array([-1,2,2.5,4,8])

    #ชุดข้อมูลสำหรับการ plot
    x_range = np.arange(-5,15,0.1)


    #model: y = a*(x**3) + b*(x**2) + c*x + d
    a = 0
    b = 0
    c = 0
    d = 0

    J_train_list = []
    J_val_list = []
    J_test_list = []
    epoch_list =[]
    epoch = 0
    
    while True:
        #prediction
        y_train = a*(x_train**3) + b*(x_train**2)+ c*x_train + d
        y_val = a*(x_val**3) + b*(x_val**2) + c*(x_val) + d
        y_test = a*(x_test**3) + b*(x_test**2) + c*(x_test) + d
        y_range = a*(x_range**3) + b*(x_range**2) + c*(x_range) + d
        N_train = len(y_train)
        N_val = len(y_val)
        N_test = len(y_test)

        #คำนวณ cost function
        J_train = (1/N_train)*np.sum((y_train-t_train)**2)
        J_val = (1/N_val)*np.sum((y_val-t_val)**2)
        J_test =(1/N_test)*np.sum((y_test-t_test)**2)
        J_test_list.append(J_test)
        J_val_list.append(J_val)
        J_train_list.append(J_train)
        epoch_list.append(epoch)
        epoch +=1

        print(f"cost: train = {J_train:.2f}, val= {J_val:.2f},test = {J_test:.2f}")


        plt.figure(1, figsize=(20,8))
        plt.clf()
        plt.subplot(1,2,1)
        plt.plot(x_train,t_train,'b.')
        plt.plot(x_test,t_test,'r.')
        plt.plot(x_val,t_val,'g.')
        plt.plot(x_range,y_range,'k')
        plt.legend(['train','test','val','model'])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid()
        

        plt.subplot(1,2,2)
        plt.plot(epoch_list,J_train_list,'b-')
        plt.plot(epoch_list,J_val_list,'r-')
        plt.plot(epoch_list,J_test_list,'g-')
        plt.legend(['train','val','test'])
        plt.grid()
        plt.xlabel("epoch")
        plt.ylabel("cost function")


        plt.show(block=False)
        plt.pause(0.001)

        if len(J_val_list) >= 2:
            if J_val_list[-1] > J_val_list[-2]:
                plt.show()
                break
        #คำนวณ gradient       
        g_a = (2/N_test)*np.sum((y_train-t_train)*(x_train**3))
        g_b = (2/N_test)*np.sum((y_train-t_train)*(x_train**2))
        g_c = (2/N_test)*np.sum((y_train-t_train)*(x_train**1))
        g_d =(2/N_test)*np.sum((y_train-t_train)*1)

        print(f"g_a = {g_a:.2f},g_b ={g_b:.2f}, g_c ={g_c:.2f}, g_d{g_d:.2f}")

        #ปรับค่า weight
        a = a - g_a*lr*0.001
        b = b - g_b*lr*0.1
        c = c - g_c*lr*1
        d = d - g_d*lr*10

        print(f"a = {a:.5f},b ={b:.5f}, c ={c:.5f}, d{d:.5f}")

        
    