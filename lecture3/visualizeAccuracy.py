from KNN import KNN
from datasetHandler import loadIrisDataset, splitTrainTest
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x, y = loadIrisDataset()
    x_train, y_train, x_test , y_test = splitTrainTest(x, y, test_size=0.5)
    

    list_k = []
    list_acc = []
    for k in range(1,71):
        knn = KNN(k)
        knn.train(x_train,y_train)
        z_test = np.array(knn.predict(x_test))
    
        acc = sum(z_test==y_test)*100/len(y_test)
        list_k.append(k)
        list_acc.append(acc)
    
    plt.figure(1)
    plt.plot(list_k,list_acc, 'b*-')
    plt.xlabel('k')
    plt.ylabel('acc(%')
    plt.ylim([0,100])
    plt.grid()
    plt.show()
        