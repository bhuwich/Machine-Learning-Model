from Node import Node
class ConvertTree:
    def __init__(self, root_node):
        self.parent = []
        self.node = []
        self.branch = []
        self.node_index = []
        self.node_object = []
        self.root_node = self.copyNode(root_node)
        self.labelIndex(self.root_node)
        self.convert(self.root_node)
        self.branch = self.branch[1:]
    
    def copyNode(self, node):
        node_ = Node()
        node_.name = node.name
        node_.branch = list(node.branch)
        for i in range(len(node.children)):
            node_.children.append(self.copyNode(node.children[i]))
        return node_

    def labelIndex(self, node):
        node.index = len(self.node_index)
        self.node_object.append(node)
        self.node_index.append(node.index)
        self.node.append(node.name)
        self.parent.append(0)
        self.branch.append('')
        for i in range(len(node.children)):
            self.labelIndex(node.children[i])

    def convert(self, node):
        for i in range(len(node.children)):
            child = node.children[i]
            self.parent[child.index] = node.index+1
            self.branch[child.index] = node.branch[i]
            self.convert(child)