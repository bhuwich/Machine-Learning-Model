from Node import Node
import numpy as np
from DecisionTree import DecisionTree
if __name__ == '__main__':
    f_ = "Tempurature"
    x_ = [40,48,60,72,80,90]
    t = ["No","No","Yes","Yes","Yes","No"]
    node = Node()
    x,f = node.convertNumericFeature(x_,t,f_)
    print(x)
    print(f)
