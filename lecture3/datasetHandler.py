import pandas as pd
import os
import numpy as np

def loadIrisDataset(path='./iris.csv'):
    if os.path.isfile(path):
        iris = pd.read_csv(path, header=None)
    else:
        url = 'http://archive.ics.uci.edu/ml/machine-learning-' + \
              'databases/iris/iris.data'
        iris = pd.read_csv(url, header = None)# อ่านไฟล์
        iris.to_csv(path, index= False, header =None)# เก็บไฟล์ 
    x = iris.iloc[:, :4].values
    y = iris.iloc[:, -1].values
    return x,y

def splitTrainTest(x,y, test_size=0.5):
    n_all = len(y)
    n_test = int(np.floor(n_all*test_size))
    n_train = n_all - n_test

    #shuffle
    ind_rand = np.random.permutation(n_all)
    x = x[ind_rand, :]
    y = y[ind_rand]

   
    x_train = x[:n_train, :]
    y_train = y[:n_train]
    x_test = x[n_train:, :]
    y_test = y[n_train:]
    return x_train, y_train, x_test, y_test

def printData(x, y):
    for i in range(len(y)):
        print(f"{i} {x[i]} {y[i]}")

if __name__ == '__main__':
    x, y = loadIrisDataset()
    print("-----------all data-------")
    printData(x,y)
    print()

    x_train, y_train, x_test, y_test = splitTrainTest(x, y)
    print("---------train data-------")
    print()
    print("---------test data--------")
    printData(x_test, y_test)
    print()         