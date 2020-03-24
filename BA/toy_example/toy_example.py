import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import random
import pandas as pd
import pickle

NAME ="toyexample_1"
IMSIZE = np.array([256, 256])

SCALE_FACTOR_X = 100 #pixels per distance
SCALE_FACTOR_Y = SCALE_FACTOR_X

FOCAL_LENGTH = 0.05

x0 = IMSIZE[0]/2.
y0 = IMSIZE[1]/2.

NUM_POINTS = 4

x_range = [-15,15.0]
y_range = [-15,15.0]
z_range = [1.0,3.0]

KERNEL_SIZE = 75 #odd number

USE_INTENSITY_OCTAVES = False

pad_left = int((KERNEL_SIZE -1) /2)
pad_right = int((KERNEL_SIZE +1) /2)

#use fixed point cloud -> Set points_3d = point_cloud in __main__
point_cloud = np.array(
    [
    [0,4,-3,5],
    [0,1.9,-0.0, -6.0],
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

def create_image_matrix(points_2d, image_suffix= ""):
    grayscale = np.zeros(shape=(IMSIZE))
    coords = points_2d.transpose() #* np.array([SCALE_FACTOR_X, SCALE_FACTOR_Y])
    #print(coords)
    coords = np.around(coords)
    coords = coords.astype(int) -1
    coords=np.flip(coords, axis=1) # coordinate order is (y_index, x_index)!
    mask = (coords[:,0] > pad_left) & (coords[:,0] < IMSIZE[0]-pad_right)
    mask = mask & (coords[:,1] > pad_left) & (coords[:,1] < IMSIZE[1]-pad_right)

    coords=coords[mask]

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
    plt.imsave("data/"+NAME + image_suffix +".png", grayscale, cmap="gray")
    #plt.show() #remove comment if you want to see image
    return mask
    

if __name__ == "__main__":
    
    points_3d = init_3d_points()
    points_2d = project_3d_points_to_image(P_DEFAULT,points_3d)
    #P = np.dot(K,np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[1.0,1.0,1.0,0.0]]))
    #points_2d_2 = project_3d_points_to_image(P,points_3d)
    mask = create_image_matrix(points_2d)
    #coords_2 = create_image_matrix(points_2d_2, image_suffix="_2")
    df = pd.DataFrame(columns=["X","Y","Z","x","y","found"], index = range(len(mask)))
    df.loc[:,["X","Y","Z"]] = points_3d[:3,:].T
    df.loc[:,["x","y"]] = points_2d.T
    df.loc[:,"found"] = mask

    np.save("data/"+NAME+"_K.npy", K) # K = np.load("data/"+NAME+"_K.npy")
    np.save("data/"+NAME+"_P.npy", P_DEFAULT) # P = np.load("data/"+NAME+"_P.npy")
        
    df.to_csv("data/"+NAME+"_data.csv", sep = ";")

    data_final = {'K':K, 'T_matrix':P_DEFAULT, '3d_points':points_3d[:3,:].T[mask], 
                    '2d_points':points_2d.T[mask]}
    pickle.dump(data_final, open("data/" + NAME + "_data.p", "wb"))

    
        

