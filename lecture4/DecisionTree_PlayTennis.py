from DecisionTree import DecisionTree
from Node import Node
from datasetHandler import loadPlayTennisDataset

if __name__ == '__main__':
    x, y, f = loadPlayTennisDataset()
    node = Node()
    node.train(x,y,f)

    #pruning by human
    node_wind = node.children[node.branch.index("Rain")]
    node_wind.branch = []
    node_wind.children = []
    node_wind.name = "No"
    node_wind.major_class = "No"

    node_o = node.children[node.branch.index("Overcast")]
    node_o.name = "Wind"
    node_o.major_class = "Yes"
    node_o.branch = ["Strong","Weak"]
    node_o.children = [Node("No"), Node("Yes")]
    



    tree = DecisionTree(root_node=node)
    tree.show()



    #ทดสอบการใช้งาน
    x = [['Rain','Hot','High','Weak'],['Sunny','Cool','High','Strong']]
    f = ['Outlook', 'Temperature', 'Humidity','Wind']
    print(tree.predict(x,f))