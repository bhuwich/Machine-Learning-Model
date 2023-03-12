from datasetHandler import loadHeartDisease,splitTrainTest, KFoldCV
from DecisionTree import DecisionTree
import numpy as np
from Node import Node
if __name__ == '__main__':
    k = 1
    x,y,f = loadHeartDisease()
    kfold_cv = KFoldCV(x,y, k)
    kfold_cv.splitKFold()

    n_correct = 0
    for i in range(k):
        print(f"evaluate fold {i} ...",end="")
        x_train, y_train, x_val, y_val = splitTrainTest(x,y, test_size =0.3)
        is_numeric = [True, False, False, True, True, False, False, True, False, True, False]

        node = Node()
        node.train(x_train,y_train,f, is_numeric)
        
    
        z_test = np.array(node.predict(x_train,f))
        print(z_test,y_val)
        n_correct_fold = np.sum(z_test==y_val)
        print(f", n= {len(y_val)}, n_correct={n_correct_fold}, acc ={n_correct_fold*100/len(y_val):.2f}%")
        n_correct += n_correct_fold
    print(f"n_all = {len(y)}, n_correct ={n_correct}, acc= {n_correct*100/len(y):.2f}%")