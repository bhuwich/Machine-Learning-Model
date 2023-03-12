from datasetHandler import loadHeartDisease,splitTrainTest, KFoldCV
from DecisionTree import DecisionTree
import numpy as np
from Node import Node
if __name__ == '__main__':
    k = 20
    x,y,f = loadHeartDisease()
    kfold_cv = KFoldCV(x,y, k)
    kfold_cv.splitKFold()
    print(x,y,f)

    n_correct = 0
    for i in range(k):
        print(f"evaluate fold {i} ...",end="")
        x_train, y_train, x_test, y_test = kfold_cv.getData(i)
        x_train, y_train, x_val, y_val = splitTrainTest(x,y, test_size =0.3)
        is_numeric = [True, False, False, True, True, False, False, True, False, True, False]

        node = Node()
        node.train(x_train,y_train,f, is_numeric)
        node.postPruning(x_val,y_val, f)
    
        z_test = np.array(node.predict(x_test,f))
        n_correct_fold = np.sum(z_test==y_test)
        print(f", n= {len(y_test)}, n_correct={n_correct_fold}, acc ={n_correct_fold*100/len(y_test):.2f}%")
        n_correct += n_correct_fold
    print(f"n_all = {len(y)}, n_correct ={n_correct}, acc= {n_correct*100/len(y):.2f}%")
    tree = DecisionTree(root_node=node)
    tree.show()    
    