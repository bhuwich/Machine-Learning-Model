import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # คำสั่ง np.random.randn() เป็นคำสั่งในการสุ่มตัวเลข โดยมีการกระจายตัวแบบ normal distribution
    # โดยมีค่าเฉลี่ยอยู่ที่ 0 และ ส่วนเบี่ยงเบนมาตรฐาน SD อยู่ที่ 1 ยิ่งจำนวนสุ่มเยอะค่า เฉลี่ยเข้าใกล้ 0 std เข้าใกล้ 1 
    # คำสั่ง np.random.rand() เป็นคำสั่งการสุ่มตัวเลข โดยมีการกระจายตัวแบบ uniform distribution
    # โดยมีช่วง 0-1


    N = 100000
    randn_data = np.random.randn(N)*3
    rand_data = np.random.rand(N)*3+1

    bins = np.arange(-10,10.1, 0.1)
    freq_randn_data, _ = np.histogram(randn_data, bins)
    freq_rand_data, _ = np.histogram(rand_data, bins)

    new_bins = (bins[0:-1] + bins[1:])/2

    print("randn_data = ")
    print(randn_data)
    print()
    print("rand_data = ")
    print(rand_data)
    print()
    print("bins = ")
    print(bins)
    print()
    print("new_bins = ")
    print(new_bins)
    print()
    print("freq_randn_data = ")
    print(freq_randn_data)
    print()
    print("freq_rand_data = ")
    print(freq_rand_data)
    print()

    plt.figure(1)
    plt.subplot(2,1,1)
    plt.title("Uniform distribution")
    plt.plot(new_bins,freq_rand_data)
    plt.ylabel("freq")
    plt.grid()
    plt.subplot(2,1,2)
    plt.title("Normal distribution")
    plt.plot(new_bins, freq_randn_data)
    plt.ylabel("freq")
    plt.xlabel("new_bins")
    plt.grid()
    plt.show()
    