import matplotlib.pyplot as plt
import numpy as np

def entropy(t):
    # เช่น t = ['cat','cat','dog','dog','cat','bird'], entropy = ?
    t = np.array(t)
    target = np.unique(t)
    n = len(t)
    e = 0
    for t_ in target:
        n_ = np.sum(t == t_)
        p = n_/n
        e_ = -p*np.log2(p) if p > 0 else 0
        e += e_
    return e



if __name__ == "__main__":

    t = ['cat','cat','dog','dog','cat','dog']
    print(entropy(t))
    ph = np.arange(0.0, 1.01, 0.01)
    pt = 1 - ph
    entropy = -ph*np.log2(ph) - pt*np.log2(pt)
    entropy[np.isnan(entropy)] = 0
    plt.figure(1)
    plt.plot(ph,entropy)
    plt.title('toss a coin')
    plt.xlabel('probability')
    plt.ylabel('entropy')
    plt.grid()
    plt.show()