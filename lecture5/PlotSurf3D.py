from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
def computeSurfData(feature, target, start_x, stop_x, res_x, start_y, stop_y, res_y, zFunc):
    x = np.arange(start_x,stop_x,res_x)
    y = np.arange(start_y,stop_y,res_y)
    z = np.zeros([x.shape[0], y.shape[0]])
    for i in range(len(x)):
        for j in range(len(y)):
            z[i,j] = zFunc(feature, target, x[i], y[j])
    y, x = np.meshgrid(y, x)
    return x, y, z

def plotSurf(ax, x, y, z, curr_x=None, curr_y=None, curr_z=None, alpha=0.3, curr_color='r', s=50,
             xlabel='X', ylabel='Y', zlabel='Z', azimuth=-60, elevation=30):
    #ax = fig.add_subplot(1,2,2, projection = '3d')
    ax.plot_surface(x, y, z, cmap=cm.cool, linewidth=0, antialiased=False, alpha = alpha)
    if curr_x is not None and curr_y is not None and curr_z is not None:
        ax.scatter(curr_x, curr_y, curr_z, color=curr_color, s=s)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.view_init(elevation, azimuth)