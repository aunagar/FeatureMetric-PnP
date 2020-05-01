import pickle
import pandas as pd

import numpy as np

import torch


if __name__=="__main__":
    name = "run2"
    df = pd.read_csv("/cluster/scratch/plindenbe/"+name+"/summary.csv", sep=";")
    
    index = 2

    query_image = df.loc[index,"query_image_origin"]
    pickle_path = df.loc[index,"track_pickle_path"]

    res_dict = pickle.load(open(pickle_path,"rb")) #Change

    #img = cv2.imread(query_image) #Change

    n_iters = 50


    l_id = 0

    
    
    

    point_list=torch.stack(res_dict["points2d"][:n_iters])
    R_list = res_dict["Rs"][:n_iters]
    cost_list = res_dict["costs"][:n_iters]
    t_list = res_dict["ts"][:n_iters]
    inlier_masks=res_dict["mask"][:n_iters]
    threshold_mask_list=res_dict["threshold_mask"][:n_iters]


    for i in range(n_iters):
        if threshold_mask_list[i] is not None:
            inlier_masks[i][inlier_masks[i]]=threshold_mask_list[i]
        if i == 0:
            continue;
        t_list[i] =t_list[i] - t_list[0] #Reset initial offset to origin -> better visualization
    t_list[0] = t_list[0] - t_list[0]

    #print(point_list[l_id, :].shape)
    #print(inlier_masks[l_id].shape)
    #mask = inlier_masks[l_id]
    #print(mask)
    #mask[mask] = threshold_mask_list[l_id]
    #print(mask)
    #print(point_list[l_id,mask,:].shape)
    pts = np.array(point_list,dtype=np.int32)
    for j,p in enumerate(pts[0]):
        print(j,p)
    print(pts[0].shape)