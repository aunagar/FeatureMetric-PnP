from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np 
import matplotlib.pyplot as plt
import math

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

def fig2data (fig):
    """
    @brief Convert a Matplotlib figure to a 3D numpy array with RGB channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGB values
    """
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGB buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h,w,3)
 
    # canvas.tostring_argb give pixmap in RGB mode.
    buf = np.roll( buf, 2, axis = 2 )
    return buf

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def cuboid_data(center, size):
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(center, size)]
    # get the length, width, and height
    l, w, h = size
    x = np.array([[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]])  # x coordinate of points in inside surface
    y = np.array([[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
         [o[1], o[1], o[1], o[1], o[1]],          # y coordinate of points in outside surface
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]])    # y coordinate of points in inside surface
    z = np.array([[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]])                # z coordinate of points in inside surface
    return x, y, z


def plot_coordinate_system(ax1,center=(0,0,0),size=(1,1,1)):
    X, Y, Z = cuboid_data(center, size)
    #ax1.plot_surface(X, Y, Z, color='b', rstride=1, cstride=1, alpha=0.1)
    ax1.set_xlabel('X')
    #ax1.set_xlim(-1, 1)
    ax1.set_ylabel('Y')
    #ax1.set_ylim(-1, 1)
    ax1.set_zlabel('Z')
    #ax1.set_zlim(-1, 1)

    # Here we create the arrows:
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)

    a = Arrow3D([0, size[0]], [0, 0], [0, 0], **arrow_prop_dict, color='r')
    ax1.add_artist(a)
    a = Arrow3D([0, 0], [0, size[1]], [0, 0], **arrow_prop_dict, color='b')
    ax1.add_artist(a)
    a = Arrow3D([0, 0], [0, 0], [0, size[2]], **arrow_prop_dict, color='g')
    ax1.add_artist(a)

    # Give them a name:
    ax1.text(0.0, 0.0, -0.1, r'$0$')
    ax1.text(1.1*size[0], 0, 0, r'$x$')
    ax1.text(0, 1.1*size[1], 0, r'$y$')
    ax1.text(0, 0, 1.1*size[2], r'$z$')

# vertices of a pyramid

def plot_camera(ax,R,t, scale =1.0, edgecolor="r", facecolor="cyan", alph=0.25):
    dz = 1
    dx = 0.3
    dy = 0.3
    v = np.array([[-dx, -dy, dz], [dx, -dy, dz], [dx, dy, dz],  [-dx, dy, dz], [0, 0, 0]]) * scale
    v = np.dot(R,v.T).T + t
    ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])

    # generate list of sides' polygons of our pyramid
    verts = [ [v[0],v[1],v[4]], [v[0],v[3],v[4]],
    [v[2],v[1],v[4]], [v[2],v[3],v[4]], [v[0],v[1],v[2],v[3]]]

    # plot sides
    ax.add_collection3d(Poly3DCollection(verts, 
    facecolors=facecolor, linewidths=1, edgecolors=edgecolor, alpha=alph))
