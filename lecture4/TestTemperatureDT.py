from Node import Node
from DecisionTree import DecisionTree
if __name__ == '__main__':
    node = Node()
    f = ['Temperature']
    x = [[40],[48],[60],[72],[80],[90]]
    t = ['No','No','Yes','Yes','Yes','No']
    is_numeric = [False]

    node.train(x,t,f,is_numeric)
    tree = DecisionTree(root_node=node)
    tree.show()