import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib.gridspec import GridSpec
import seaborn as sns
import cv2
import pandas as pd
import matplotlib.patches as patches
import sys
import pickle
from matplotlib.patches import ConnectionPatch

sys.path.append('s2dhm/') #Should be autodetected later in __init__.py file!
sys.path.append('featurePnP/')
sys.path.append('visualization/')
from datasets import dataload_helpers as data_helpers
def load_data(triangulation_filepath, nvm_filepath, filenames):
    filename_to_pose = data_helpers.from_nvm(nvm_filepath, filenames)
    filename_to_local_reconstruction = data_helpers.load_triangulation_data(triangulation_filepath, filenames)
    filename_to_intrinsics = data_helpers.load_intrinsics(filenames)

    return filename_to_pose, filename_to_local_reconstruction, filename_to_intrinsics

def top_pdist(a, b, N):
    d = np.sum((b - a)**2, axis=0)
    return heapq.nlargest(N, range(len(d)), d.__getitem__)


from visualization import visualize_hc, plot_3d_tools, create_video
import plot_3d_tools as plot_3d

def add_subplot_border(ax, width=1, color=None ):
    
    fig = ax.get_figure()

    # Convert bottom-left and top-right to display coordinates
    x0, y0 = ax.transAxes.transform((0, 0))
    x1, y1 = ax.transAxes.transform((1, 1))

    # Convert back to Axes coordinates
    x0, y0 = ax.transAxes.inverted().transform((x0, y0))
    x1, y1 = ax.transAxes.inverted().transform((x1, y1))

    rect = plt.Rectangle(
        (x0, y0), x1-x0, y1-y0,
        color=color,
        transform=ax.transAxes,
        zorder=-1,
        lw=2*width+1,
        fill=None,
    )
    fig.patches.append(rect)

def plot_update(ax, i, rgb_img, R_list, t_list, pts, dpi=160, cost_plot_height=300, mask_inlier_list = None, 
                limits_x=None, limits_y=None, hc_image_dict=None):
    if len(pts.shape)!=2:
        #ax.clear()
        #ax.axis("off")
        frame_c = ["blue","red","orange","gray"]
        # Add the patch to the Axes
        
        k=0
        for j, hc_image in hc_image_dict.items():
            x=pts[-1,j,0] - 20
            y=pts[-1,j,1] - 20
            rect = patches.Rectangle((x,y),40,40,linewidth=1,edgecolor=frame_c[k],facecolor='none')
            ax.add_patch(rect)
            k+=1
    # Create a copy of base image to work on
    img_int = rgb_img.copy()
    # Convert image to an array so we can concatenate with our image
    
    # For every 2D point in our image (left)
    if len(pts.shape)!=2:
        ax.imshow(img_int)
        for j, p in enumerate(pts[i]):
            # Draw past line
            cv2.polylines(img_int, np.int32([pts[:i+1,j,:]]), False, (0,0,255), 1)
            # Draw start point
            cv2.circle(img_int, tuple(pts[0,j,:]), 4, (0,255,0), -1) 
            # Draw current point
            cv2.circle(img_int, tuple(pts[i,j,:]), 4, (0,0,255), -1) 
            # Draw Inliers
            if mask_inlier_list is not None and mask_inlier_list[i,j]:
                cv2.circle(img_int, tuple(pts[i,j,:]), 4, (255,0,0), -1)
    else:
        #ax.imshow(img_int, extent=[0,256,256,0])
#         # Draw past line
#         cv2.polylines(img_int, np.int32([pts[:i+1,:]/4]), False, (0,0,255), 1)
#         # Draw start point
#         cv2.circle(img_int, tuple(pts[0,:]), 4, (0,255,0), -1) 
#         # Draw current point
#         cv2.circle(img_int, tuple(pts[i,:]), 4, (0,0,255), -1) 
#         # Draw Inliers
#         if mask_inlier_list is not None and mask_inlier_list[i,j]:
#             cv2.circle(img_int, tuple(pts[i,:]), 4, (255,0,0), -1)
        ax.scatter(pts[:i,0]/4.0,pts[:i,1]/4.0, c="blue", s = i/20.0)
        # for k in range(i+1):
        #     t = list(pts[k,:]/4)
        #     t[0] = int(t[0])
        #     t[1] = int(t[1])
        #     cv2.circle(img_int,tuple(t) , 4, (0,0,255), -1) 

    # Append image list
    ax.imshow(img_int)

def create_corresp_axes():
    fig = plt.figure(figsize=(14,9), dpi = 150)

    gs0 = GridSpec(1, 2, left=0.02, right=0.98, wspace = 0.0)
    ax0 = fig.add_subplot(gs0[0])
    ax1 = fig.add_subplot(gs0[1])
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    fig.tight_layout()
    laxis = [ax0,ax1]
    for ax in laxis:
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.tick_params(labelbottom=False, labelleft=False)
    return fig, laxis


def create_grid_axes():
    fig = plt.figure(dpi=500)

    gs0 = GridSpec(2, 1, left=0.02, right=0.24)
    ax0 = fig.add_subplot(gs0[0])
    ax1 = fig.add_subplot(gs0[1])

    gs1 = GridSpec(1, 1, left=0.26, right=0.74)
    ax2 = fig.add_subplot(gs1[0])

    gs2 = GridSpec(2, 1, left=0.76, right=0.98)
    ax3 = fig.add_subplot(gs2[0])
    ax4 = fig.add_subplot(gs2[1])
    
    axes = [ax0,ax1,ax2,ax3,ax4]
    
    frame_c = ["blue","red",None,"orange","gray"]
    k=0
    for ax in axes:
        ax.axis("off")
        ax.tick_params(labelbottom=False, labelleft=False)
        add_subplot_border(ax, 2, frame_c[k])
        k=k+1
    return fig, axes

def save_hc_video(video_path, query_image, hc_image_dict, n_iters, track_dict, patch_size=10):
    point_list=torch.stack(track_dict["points2d"][:n_iters])
    R_list = track_dict["Rs"][:n_iters]
    cost_list = track_dict["costs"][:n_iters]
    t_list = track_dict["ts"][:n_iters]
    inlier_masks=track_dict["mask"][:n_iters]
    threshold_mask_list=track_dict["threshold_mask"][:n_iters]
    fig, axes = create_grid_axes()
    ax = axes[2]
    height, width, layers = plot_3d.fig2data(fig).shape
    video= cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 2, (width,height))
    
    ax_hc=[axes[0],axes[1],axes[3],axes[4]]
    pts = np.array(point_list,dtype=np.int32)
    
    k=0
    for j, hc_image in hc_image_dict.items():
        x=pts[-1,j,0]/4
        y=pts[-1,j,1]/4
        ax_hc[k].set_xlim([x-patch_size/2,x+patch_size/2])
        ax_hc[k].set_ylim([y+patch_size/2,y-patch_size/2])
        k+=1
    video= cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width,height))
    
    for i in range(n_iters):
        if threshold_mask_list[i] is not None:
            inlier_masks[i][inlier_masks[i]]=threshold_mask_list[i]
        if i == 0:
            continue;
        t_list[i] =t_list[i] - t_list[0] #Reset initial offset to origin -> better visualization
    t_list[0] = t_list[0] - t_list[0]
    
    for i in range(n_iters):
        print(i)
        plot_update(ax, i, query_image, R_list, t_list, pts, hc_image_dict = hc_image_dict)
        k=0
        for j, hc_image in hc_image_dict.items():
            plot_update(ax_hc[k], i, hc_image, R_list, t_list, pts[:,j,:])
            con = ConnectionPatch(xyA=[100,100], xyB=[100,100], coordsA="data", coordsB="data",
                      axesA=ax, axesB=ax_hc[k], color="red")
            
            ax.add_artist(con)
            k+=1  
        video.write(plot_3d.fig2data(fig))
    video.release()

if __name__=="__main__":
    df = pd.read_csv("results/subset/night/summary.csv",sep=";")
    for idx in range(df.shape[0]):
        print("Index:",idx)
        row = df.iloc[idx]
        img = cv2.imread(row.query_image_origin)#cv2.cvtColor(cv2.imread(row.reference_image_origin), cv2.COLOR_BGR2RGB)
        track = pickle.load(open(row["track_pickle_path"], "rb"))
        hc_dict=track["hc_dict"]
        print(hc_dict)
        hc_image_dict={}
        k=0
        hc_dict
        for key in hc_dict:
            hc_image_dict[key] = cv2.imread(hc_dict[key])
            k+=1
            if k==4:
                break
        print(hc_image_dict.keys())
        save_hc_video("results/subset/night/hcvideonight"+str(idx)+".mp4", img, hc_image_dict, 10, track)