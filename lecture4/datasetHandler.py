import pandas as pd
import os
import numpy as np
class KFoldCV:
    def __init__(self,x,y,k):
        self.x = x
        self.y = y
        self.k = k
        self.ind_all = []
        self.ind_split = []
    
    def splitKFold(self):
        self.ind_all = np.random.permutation(len(self.y))
        self.ind_split = np.array_split(self.ind_all,self.k)

    def getData(self,fold):
        ind_test = self.ind_split[fold]
        ind_train = np.setdiff1d(self.ind_all,ind_test)
        x_train = self.x[ind_train,:]
        y_train = self.y[ind_train]
        x_test = self.x[ind_test,:]
        y_test = self.y[ind_test]
        return x_train,y_train,x_test,y_test

def loadPlayTennisDataset(path='./iris.csv'):
    if os.path.isfile(path):
        play_tennis = pd.read_csv(path, header=None)
    else:
        url = 'https://gist.githubusercontent.com/DiogoRibeiro7/c6590d0cf119e87c39e31c21a9c0f3a8/raw/4a8e3da267a0c1f0d650901d8295a5153bde8b21/PlayTennis.csv'
        play_tennis = pd.read_csv(url, header = None)# อ่านไฟล์
        play_tennis.to_csv(path, index= False, header =None)# เก็บไฟล์ 
    x = play_tennis.iloc[1:, :-1].values
    y = play_tennis.iloc[1:, -1].values
    f = play_tennis.iloc[0,:-1].values
    return x,y,f

def loadCarEvaluation(path='./car_evaluation.csv'):
    if os.path.isfile(path):
        car = pd.read_csv(path)
    else:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
        car = pd.read_csv(url, header = None)# อ่านไฟล์
        car.to_csv(path, index= False)# เก็บไฟล์ 
    feature_names = ['buying','maint','doors','persons','lug_boot','safety']
    x = car.iloc[:, :-1].values
    y = car.iloc[:, -1].values
    f=np.array(feature_names)
    return x,y,f

def loadHeartDisease(path='./heart.csv'):
    if os.path.isfile(path):
        heart = pd.read_csv(path, header=None)
    else:
        raise ValueError("{path} is not found !!")
    x = heart.iloc[1:, :-1].values
    y = heart.iloc[1:, -1].values
    f = heart.iloc[0,:-1].values
    print(x)
    print(y)
    print(f)
    
    for i in range(len(y)):
        y[i] = 'Yes' if y[i] == '1' else 'No'
            
    return x,y,f
def splitTrainTest(x, y, test_size=0.5):
    n_all = len(y)
    n_test = int( np.floor(n_all*test_size) )

    # shuffle
    ind_rand = np.random.permutation(n_all)
    x = x[ind_rand, :]
    y = y[ind_rand]

    ind_test = np.arange(n_test)
    ind_train = np.setdiff1d( np.arange(n_all), ind_test )

    x_train = x[ind_train, :]
    y_train = y[ind_train]
    x_test = x[ind_test, :]
    y_test = y[ind_test]
    return x_train, y_train, x_test, y_test