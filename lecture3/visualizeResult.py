from KNN import KNN
from datasetHandler import loadIrisDataset, splitTrainTest
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x, y = loadIrisDataset()
    x_train, y_train, x_test , y_test = splitTrainTest(x, y, test_size=0.5)
    knn = KNN(3)
    knn.train(x_train,y_train)
    z_test = np.array(knn.predict(x_test))
    
    plt.figure(1)
    plt.plot(x_train[y_train=="Iris-setosa", 0], x_train[y_train=="Iris-setosa", 1] ,'b.')
    plt.plot(x_train[y_train=="Iris-virginica", 0], x_train[y_train=="Iris-virginica", 1] ,'r.')
    plt.plot(x_train[y_train=="Iris-versicolor", 0], x_train[y_train=="Iris-versicolor", 1] ,'g.')
    plt.plot(x_test[z_test=='Iris-setosa',0], x_test[z_test=='Iris-setosa',1], 'b*')
    plt.plot(x_test[z_test=='Iris-virginica',0], x_test[z_test=='Iris-virginica',1], 'r*')
    plt.plot(x_test[z_test=='Iris-versicolor',0], x_test[z_test=='Iris-versicolor',1], 'g*')
    plt.plot(x_test[y_test=='Iris-setosa',0], x_test[y_test=="Iris-setosa",1],'bo',mfc="none")
    plt.plot(x_test[y_test=='Iris-virginica',0], x_test[y_test=="Iris-virginica",1],'ro',mfc="none")
    plt.plot(x_test[y_test=='Iris-versicolor',0], x_test[y_test=="Iris-versicolor",1],'go',mfc="none")
    
    plt.xlabel('axis 0')
    plt.ylabel('axis 1')
    plt.grid()
    plt.show()