from DecisionTree import DecisionTree
from Node import Node
from datasetHandler import loadCarEvaluation, KFoldCV, splitTrainTest
import numpy as np

if __name__ == '__main__':
    
    #set parameter
    k = 5

    #load Data
    x,y,f = loadCarEvaluation()

    #split data

    kfold_cv = KFoldCV(x,y, k)
    kfold_cv.splitKFold()
    
    
    #processing
    n_correct = 0
    for i in range(k):
        print(f"evaluate fold {i} ...",end="")

        #get data train/test
        x_train, y_train, x_test, y_test = kfold_cv.getData(i)

        #build model
        x_train, y_train, x_val, y_val = splitTrainTest(x_train,y_train,test_size=0.3)
        node = Node()
        node.train(x_train, y_train,f) #train model
        print(x_val,y_val)
        node.postPruning(x_val,y_val, f)

        # prediction 
        z_test = np.array(node.predict(x_test,f))

        # evaluation 
        n_correct_fold = np.sum(z_test==y_test)
        
        print(f", n= {len(y_test)}, n_correct={n_correct_fold}, acc ={n_correct_fold*100/len(y_test):.2f}%")
        n_correct += n_correct_fold
    print(f"n_all = {len(y)}, n_correct ={n_correct}, acc= {n_correct*100/len(y):.2f}%")

    