import sys
import os
import cv2

sys.path.insert(0, os.path.abspath('../../../'))
from BA.visualization.create_video import *
import torch
import pickle

"""
Script to write a video from a track_pickle_file and a raw query image. Might take a few minutes.

Known problems: The opacity of camera pose cone's surfaces does not work when running the script 
on the cluster, but works locally. Probably some version problem in seaborn or matplotlib.

"""

if __name__=="__main__":
    res_dict = pickle.load(open("model_pickle.p","rb")) #Change

    img = cv2.imread("s2dhm_results/query_0_raw.png") #Change

    point_list=torch.stack(res_dict["points2d"])
    R_list = res_dict["Rs"][:50]
    cost_list = res_dict["costs"][:50]
    t_list = res_dict["ts"][:50]
    a = t_list[0]
    for i in range(1,len(t_list)):
        t_list[i] =t_list[i] - t_list[0] #Reset initial offset to origin -> better visualization
    t_list[0] = t_list[0] - t_list[0]
    frames = create_frames_with_camera_pose(img, R_list, t_list,cost_list, point_list)


    save_video(frames,"full_video.mp4") #Change