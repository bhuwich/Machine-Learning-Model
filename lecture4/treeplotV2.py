from treeplot import treeplot as treeplotV1
from ConvertTree import ConvertTree

def treeplot(tree, show=True, fig_no=None, note_style='ob', edge_style='c', block=True):
    ct = ConvertTree(tree.root_node)
    parent = ct.parent
    vlabel = ct.node
    elabel = ct.branch
    treeplotV1(tree=parent, vlabel=vlabel, elabel=elabel, show=True, fig_no=fig_no,
             note_style='ob', edge_style='c',block=block)