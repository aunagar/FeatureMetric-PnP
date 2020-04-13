import sys
import os
import cv2

sys.path.insert(0, os.path.abspath('../../../'))
from BA.visualization.create_video import *
import torch
import pickle

if __name__=="__main__":
    res_dict = pickle.load(open("model_pickle.p","rb"))

    img = cv2.imread("s2dhm_results/query_0_raw.png")

    point_list=torch.stack(res_dict["points2d"])
    R_list = res_dict["Rs"][:50]
    cost_list = res_dict["costs"][:50]
    t_list = res_dict["ts"][:50]
    a = t_list[0]
    for i in range(1,len(t_list)):
        t_list[i] =t_list[i] - t_list[0]
        print(t_list[i])
    t_list[0] = t_list[0] - t_list[0]
    frames = create_frames_with_camera_pose(img, R_list, t_list,cost_list, point_list)


    save_video(frames,"full_video.mp4")