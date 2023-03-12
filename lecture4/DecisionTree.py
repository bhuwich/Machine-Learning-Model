from Node import Node
from treeplotV2 import treeplot

class DecisionTree:
    def __init__(self, root_node=Node()):
        self.root_node = root_node
    

    def predict(self, x, f=None):
        return self.root_node.predict(x, f)
    
    def show(self, fig_no=None, block=True):
        treeplot(self, fig_no=fig_no,block=block)
