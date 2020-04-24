import sys
import os
import cv2
import pandas as pd
sys.path.insert(0, os.path.abspath('../../../'))
from BA.visualization.create_video import *
import torch
import pickle

"""
Script to write a video from a track_pickle_file and a raw query image. Might take a few minutes.

Known problems: The opacity of camera pose cone's surfaces does not work when running the script 
on the cluster, but works locally. Probably some version problem in seaborn or matplotlib.

bsub -W 1:00 -n 1 -R "rusage[mem=32768]" -I python video.py

"""

if __name__=="__main__":
    name = "run1"
    df = pd.read_csv("/cluster/scratch/plindenbe/"+name+"/summary.csv", sep=";")
    
    index = 30

    query_image = df.loc[index,"query_image_origin"]
    pickle_path = df.loc[index,"track_pickle_path"]

    res_dict = pickle.load(open(pickle_path,"rb")) #Change

    img = cv2.imread(query_image) #Change

    n_iters = 50

    
    
    

    point_list=torch.stack(res_dict["points2d"][:n_iters])
    R_list = res_dict["Rs"][:n_iters]
    cost_list = res_dict["costs"][:n_iters]
    t_list = res_dict["ts"][:n_iters]
    inlier_masks=res_dict["mask"][:n_iters]
    threshold_mask_list=res_dict["threshold_mask"][:n_iters]

    print(t_list[-1])
    print(R_list[-1])
    print(point_list[-1])
    print(torch.stack(inlier_masks)[])
    exit()

    #exit()

    for i in range(n_iters):
        if threshold_mask_list[i] is not None:
            inlier_masks[i][inlier_masks[i]]=threshold_mask_list[i]
        if i == 0:
            continue;
        t_list[i] =t_list[i] - t_list[0] #Reset initial offset to origin -> better visualization
    t_list[0] = t_list[0] - t_list[0]
    frames = create_frames_with_camera_pose(img, R_list, t_list,cost_list, point_list, mask_inlier_list=torch.stack(inlier_masks) if threshold_mask_list[0] is not None else None)


    save_video(frames,name+"_"+str(index)+".mp4") #Change