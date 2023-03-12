import numpy as np
import matplotlib.pyplot as plt

def plotHyperPlane2d(x, t, w, b, c1='1', c2='2'):
    w = np.append(w,b)
    x = np.array(x)
    t = np.array(t)
    plt.plot(x[t==c1,0], x[t==c1,1],'og')
    plt.plot(x[t==c2,0], x[t==c2,1], 'or')
    if w[1] != 0:
        xlim = plt.gca().get_xlim()
        slope = -w[0] / w[1]
        bias = 0
        if len(w) > 2:
            bias = -w[2]/w[1]
        plt.plot(xlim, [xlim[0]*slope+bias, xlim[1]*slope+bias], 'b')
    else:
        ylim = plt.gca().get_ylim()
        plt.plot([0,0],ylim,'b')