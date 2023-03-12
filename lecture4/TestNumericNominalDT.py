from Node import Node
from DecisionTree import DecisionTree
if __name__ == '__main__':
    node = Node()
    f = ['Age','Sex','Eat Pizza']
    x = [[25,'Male','Yes'],
        [35,'Female','No'],
        [40,'Female','Yes'],
        [50,'Male','Yes'],
        [60,'Female','Yes']
        ]
    t = ['Fat','Slim','Fat','Fat','Slim']
    isNumeric = [False, False, False]
    node.train(x,t,f,isNumeric)
    tree = DecisionTree(node)
    tree.show()
        