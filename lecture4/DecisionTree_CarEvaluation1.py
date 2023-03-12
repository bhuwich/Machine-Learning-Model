from DecisionTree import DecisionTree
from Node import Node
from datasetHandler import loadCarEvaluation, splitTrainTest
import numpy as np

if __name__ == '__main__':

    

    #load data
    x,y,f = loadCarEvaluation()

    #split train test
    x_train, y_train, x_test, y_test = splitTrainTest(x,y, test_size =0.3)
    

    #train node 
    node = Node()
    node.train(x_train, y_train ,f)
    node.postPruning(x_test, y_test, f)

    
    # build decision tree
    tree = DecisionTree(root_node=node)
    tree.show()