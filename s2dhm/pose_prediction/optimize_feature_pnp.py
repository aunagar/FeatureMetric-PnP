
import gin
import os, sys
import numpy as np
import torch

from pose_prediction import matrix_utils

# this is a hack
# sys.path.insert(0, os.path.abspath('../../'))

from model import sparseFeaturePnP, find_inliers
from helpers.utils import sobel_filter

def feature_pnp_multi(query_hypercolumns, reference_hypercolumns, prediction, K, image_shape, track = False):
    relative_shape = np.array([reference_hypercolumns.shape[2] / image_shape[0], reference_hypercolumns.shape[3] / image_shape[1]])
    pts3D = torch.from_numpy( prediction.points_3d.reshape(-1,3) ) 
    ref2d = torch.from_numpy(relative_shape).view(1,2) * torch.from_numpy(prediction.reference_inliers)
    ref2d = torch.flip( ref2d.type(torch.IntTensor), (1,) )

    feature_ref = torch.cat([reference_hypercolumns.squeeze(0)[:, i, j].unsqueeze(0) for i, j in zip(ref2d[:,0], ref2d[:,1])]).type(torch.DoubleTensor)
    feature_map_query = query_hypercolumns.squeeze(0).type(torch.DoubleTensor)
    # T_init = filename_to_pose['/'.join(ref_images[0].split('/')[-3:])][1]
    T_init = prediction.matrix
    R, t = torch.from_numpy(T_init[:3, :3]), torch.from_numpy(T_init[:3,3])
    feature_grad_x, feature_grad_y = sobel_filter(feature_map_query)
    
    
    if prediction.inlier_mask is not None:
        mask_array = np.zeros(pts3D.shape[0])
        mask_array[prediction.inlier_mask]=1
        inliers = torch.from_numpy(mask_array).type(torch.BoolTensor)
    else:
        print("No initial inliers found.")
        inliers = find_inliers(pts3D, R, t, feature_map_query, feature_ref, K,image_shape[0], image_shape[1])

    model = sparseFeaturePnP() #Parameters from gin!
    for i in range(3):
        R, t = model(pts3D[inliers,:], feature_ref[inliers,:], feature_map_query, feature_grad_x, feature_grad_y,
        K, image_shape[0], image_shape[1], R_init=R, t_init=t, track = track) #Remove if no more needed
        inliers = find_inliers(pts3D, R, t, feature_map_query, feature_ref, K,image_shape[0], image_shape[1])
    return R, t, model


def feature_pnp(query_hypercolumns, reference_hypercolumns, prediction, K, image_shape, track = False, feature_pyramid = None):
    relative_shape = np.array([reference_hypercolumns.shape[2] / image_shape[0], reference_hypercolumns.shape[3] / image_shape[1]])
    pts3D = torch.from_numpy( prediction.points_3d.reshape(-1,3) ) 
    ref2d = torch.from_numpy(relative_shape).view(1,2) * torch.from_numpy(prediction.reference_inliers)
    ref2d = torch.flip( ref2d.type(torch.IntTensor), (1,) )

    feature_ref = torch.cat([reference_hypercolumns.squeeze(0)[:, i, j].unsqueeze(0) for i, j in zip(ref2d[:,0], ref2d[:,1])]).type(torch.DoubleTensor)
    feature_map_query = query_hypercolumns.squeeze(0).type(torch.DoubleTensor)
    # T_init = filename_to_pose['/'.join(ref_images[0].split('/')[-3:])][1]
    T_init = prediction.matrix
    R, t = torch.from_numpy(T_init[:3, :3]), torch.from_numpy(T_init[:3,3])
    feature_grad_x, feature_grad_y = sobel_filter(feature_map_query)
    
    model = sparseFeaturePnP() #Parameters from gin!
    if feature_pyramid is None:
        R, t = model(pts3D, feature_ref, feature_map_query, feature_grad_x, feature_grad_y,
        K, image_shape[0], image_shape[1], R_init=R, t_init=t, track = track) #Remove if no more needed
    else:
        R, t = model.multilevel_optimization(feature_pyramid, pts3D, feature_ref, feature_map_query, feature_grad_x, feature_grad_y,
        K, image_shape[0], image_shape[1], R_init=R, t_init=t, track = track)
    
    return R, t, model

@gin.configurable
def optimize_feature_pnp(query_hypercolumns, net, prediction, K, image_shape, track = False, feature_pyramid = None):
    K = torch.from_numpy( K )
    print("Initial : {}".format(list(prediction.quaternion) + list(prediction.matrix[:3,3])))

    reference_hypercolumns, _ = net.compute_hypercolumn( [prediction.reference_filename], to_cpu=False, resize=True )

    R, t, model = feature_pnp(query_hypercolumns, reference_hypercolumns, prediction, K, image_shape, track = track, feature_pyramid = feature_pyramid)
    
    T = np.eye(4)
    T[:3, :3], T[3,:3] = R.numpy(), t.numpy()

    quaternion = matrix_utils.matrix_quaternion(T)
    print("Final : {}".format(list(quaternion) + list(t.numpy())))
    return list(t.numpy()), list(quaternion), model #Remove if no more needed
