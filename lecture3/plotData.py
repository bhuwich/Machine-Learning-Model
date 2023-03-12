import matplotlib.pyplot as plt
from datasetHandler import loadIrisDataset
from datasetHandler import splitTrainTest

if __name__ == '__main__':
    x,y = loadIrisDataset()
    x_train, y_train, x_test, y_test =splitTrainTest(x, y, test_size=0.5)
    plt.plot(x_train[y_train=="Iris-setosa", 0], x_train[y_train=="Iris-setosa", 1] ,'b.')
    plt.plot(x_train[y_train=="Iris-virginica", 0], x_train[y_train=="Iris-virginica", 1] ,'r.')
    plt.plot(x_train[y_train=="Iris-versicolor", 0], x_train[y_train=="Iris-versicolor", 1] ,'g.')
    plt.plot(x_test[:, 0], x_test[:, 1], 'k.')
    plt.legend(['setosa', 'verginica', 'versicolor', 'test'])
    
    plt.xlabel('axis 0')
    plt.ylabel('axis 1')
    plt.grid()
    plt.show()