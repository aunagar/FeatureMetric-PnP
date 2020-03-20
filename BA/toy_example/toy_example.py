import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import random
import pandas as pd
NAME ="toyexample_1"
IMSIZE = np.array([64, 64])

SCALE_FACTOR_X = 10 #pixels per distance
SCALE_FACTOR_Y = SCALE_FACTOR_X

FOCAL_LENGTH = 0.05

x0 = IMSIZE[0]/2./SCALE_FACTOR_X 
y0 = IMSIZE[1]/2./SCALE_FACTOR_Y

NUM_POINTS = 6

x_range = [-7.0,7.0]
y_range = [-7.0,7.0]
z_range = [1.0,3.0]

KERNEL_SIZE = 25 #odd number

USE_INTENSITY_OCTAVES = True

pad_left = int((KERNEL_SIZE -1) /2)
pad_right = int((KERNEL_SIZE +1) /2)

#use fixed point cloud -> Set points_3d = point_cloud in __main__
point_cloud = np.array(
    [
    [2,4,-3,5],
    [-1.0,1.9,-0.0, -6.0],
    [1.0,1.0,2.0,2.0],
    [1.0,1.0,1.0,1.0]
    ]
)

K = np.array([[FOCAL_LENGTH*SCALE_FACTOR_X,0.0,x0],[0.0,FOCAL_LENGTH*SCALE_FACTOR_Y,y0],[0.0,0.0,1.0]])

P_DEFAULT = np.dot(K,np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0]]))

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)

    return kern2d/kern2d.max()

def init_3d_points():
    return np.array([[random.uniform(*x_range),random.uniform(*y_range),random.uniform(*z_range),1.0] for i in range(NUM_POINTS)]).T

def project_3d_points_to_image(proj_matrix, points):
    inhomogeneous = np.dot(proj_matrix,points)
    homogenous = inhomogeneous / inhomogeneous[2]
    return homogenous [:2]

def visualize_scene():
    return 0

def create_image_matrix(points_2d):
    grayscale = np.zeros(shape=(IMSIZE))
    coords = points_2d.transpose() * np.array([SCALE_FACTOR_X, SCALE_FACTOR_Y])
    coords = np.around(coords)
    coords = coords.astype(int) -1
    coords=np.flip(coords) # coordinate order is (y_index, x_index)!

    coords=coords[(coords[:,0] > pad_left) & (coords[:,0] < IMSIZE[0]-pad_right)]
    coords=coords[(coords[:,1] > pad_left) & (coords[:,1] < IMSIZE[1]-pad_right)]

    #print(coords.T)
    lost_points = points_2d.shape[1]-coords.shape[0]

    if lost_points > 0:
        print("%d points are ignored since they lye outside the image." % lost_points)
    #Clip
    kernel = gkern(kernlen=KERNEL_SIZE)
    if USE_INTENSITY_OCTAVES:
        (maximas, step) = np.linspace(255.0,0.0,coords.shape[0], endpoint=False, retstep=True)
        grayscale[tuple(coords.T)] = maximas 
    else:
        step = 255.0
        grayscale[tuple(coords.T)] = 255.0
    
    for point in coords:
        patch = grayscale[point[0]-pad_left:point[0]+pad_right,
        point[1]-pad_left:point[1]+pad_right]
        grayscale[point[0]-pad_left:point[0]+pad_right,
        point[1]-pad_left:point[1]+pad_right]=np.maximum(patch,grayscale[tuple(point)] * kernel)
    
    plt.imshow(grayscale,'gray')
    plt.imsave(NAME + ".png", grayscale, cmap="gray")
    #plt.show() #remove comment if you want to see image
    return coords
    

if __name__ == "__main__":
    points_3d = init_3d_points()
    points_2d = project_3d_points_to_image(P_DEFAULT,points_3d)
    coords = create_image_matrix(points_2d)

    with open("data/" + NAME+"_data.txt","w") as file: # Use file to refer to the file object
        file.write("K:\n")
        file.write(K.__repr__())
        file.write("\nP_DEFAULT:\n")
        file.write(P_DEFAULT.__repr__())
        file.write("\n3D points:\n")
        file.write(points_3d.__repr__())
        file.write("\n2D points:\n")
        file.write(points_2d.__repr__())
        file.write("\n2D pixel coordinates:\n")
        file.write(coords.__repr__())
        

