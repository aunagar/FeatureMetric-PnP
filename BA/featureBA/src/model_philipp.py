from __future__ import print_function, division

from .utils import (from_homogeneous, to_homogeneous,
                batched_eye_like, skew_symmetric, so3exp_map)

from .utils import squared_loss, scaled_loss
import torch
from torch import nn
import numpy as np

"""
@ToDo:
    - Check if this masking of supported points works as expected
    - We project points 2 times, should decrease to once only (perf.)
    - Function returns the last R and t, which are not necessarily the best
    - Try to explain this spurious oscillations sometimes
    - Try to find a workaround for local minimas, e.g. perturb randomly around best guess
    - Penalize removing points with mask (should add to error)
    - Threshold far-off feature errors
"""

"""
C...Number of Channels (= size of feature descriptor)
N...Number of Features
H...Height of the image (pixels)
W...Width of the image (pixels)
"""

def optimizer_step(g, H, lambda_=0, lr = 1.):
    """One optimization step with Gauss-Newton or Levenberg-Marquardt.
    Args:
        g: batched gradient tensor of size (..., N).
        H: batched hessian tensor of size (..., N, N).
        lambda_: damping factor for LM (use GN if lambda_=0).
    """
    if lambda_:  # LM instead of GN
        D = (H.diagonal(dim1=-2, dim2=-1) + 1e-9).diag_embed()
        H = H + D*lambda_
    try:
        P = torch.inverse(H)
    except RuntimeError as e:
        logging.warning(f'Determinant: {torch.det(H)}')
        raise e
    delta = -lr * (P @ g[..., None])[..., 0] # generally learning rate does not make sense for second order methods
    return delta

def indexing_(feature_map, points):
    '''
    Function gives x and y gradients for 3D points in camera frame.

    inputs: (All pytorch tensors)
    @feature_map : x gradient of the feature map (CxHxW)
    @points : pixel coordinates of points (Nx2)

    outputs: 
    features : features for the points (NxC)
    '''
    C,H,W = feature_map.shape
    
    features = []
    for i,j in zip(points[:,0], points[:,1]):
        features.append(feature_map[:, i, j].unsqueeze(0))

    features = torch.cat(features)

    return features

def points_within_image(points_2d, width, height, pad=0):
    '''
    Function gives a mask for points_2d that only returns points that are within width and height.

    inputs:
    @points_2d : pixel coordinates of points (Nx2) (pytorch tensor)
    @width : Width of image in pixels
    @height : Height of image in pixels
    @pad : Offset from width and height (defaults to zero)

    outputs: 
    mask : Flag whether each point is within the range (Nx1)
    '''

    mask = (points_2d[:,0] >= pad) & \
           (points_2d[:,1] >= pad) & \
           (points_2d[:,0] < width - pad) & \
           (points_2d[:,1] < height - pad)

    return mask


class sparse3DBA(nn.Module):
    def __init__(self, n_iters, loss_fn = squared_loss, lambda_ = 0.01,
                opt_depth=True, verbose=False):
        super().__init__()
        self.iterations = n_iters
        self.loss_fn = loss_fn
        self.verbose = verbose
        self.lambda_ = lambda_
        self.track_ = {"Rs":[],"ts":[], "costs": [], "points2d":[], "points3d":[], "mask":[]}


    def track(self, R,t, cost, points_2d, points_3d, mask):
        self.track_["Rs"].append(R)
        self.track_["ts"].append(t)
        self.track_["costs"].append(cost)
        self.track_["points2d"].append(points_2d)
        self.track_["points3d"].append(points_3d)
        self.track_["mask"].append(mask)

    def forward(self, pts3D, feature_ref, feature_map_query,
                feature_grad_x, feature_grad_y, K, R_init=None, t_init=None,
                confidence=None, scale=None, track = False, log = True):
        '''
        inputs:
        @pts3D : 3D points in reference camera frame (Nx3)
        @feature_ref: features for these 3D points in reference frame (NxC)
        @feature_map_query: feature map for the query image (CxHxW)
        @feature_grad_x : feature gradient map for query image (CxHxW)
        @feature_grad_y : feature gradient map for query image (CxHxW)
        @K: Camera matrix
        '''

        # This might be convenient but might to lead to errors
        if R_init is None: # If R is not inialized, initialize it as identity
            R = torch.eye(3).to(pts3D)
        else:
            R = R_init
        if t_init is None:
            t = pts3D.new_tensor([1, 1, 0]).float()
        else:
            t = t_init

        # regularization parameter
        lambda_ = self.lambda_ # lambda for LM method
        lr = 0.1 # learning rate
        lr_reset = 0.1 #reset learning rate
            

        for i in range(self.iterations):

            # project all points using current R and T on image
            points_3d = torch.mm(R, pts3D.T).T + t
            points_2d = torch.round(from_homogeneous(torch.mm(K, points_3d.T).T)).type(torch.IntTensor)-1

            # Get the points that are supported within our image size
            layers, height, width  = feature_map_query.shape
            mask_supported = points_within_image(points_2d,height, width)

            # We only take supported points
            points_2d_supported = points_2d[mask_supported,:]
            points_3d_supported = points_3d[mask_supported,:]

            # Feature error, NxC
            if self.verbose:
                print("Supported mask: ",mask_supported)
                print("All points: ", points_2d)
            error = indexing_(feature_map_query, torch.flip(points_2d_supported,(1,))) - feature_ref[mask_supported]

            # Cost of Feature error -> SSD
            cost = 0.5*(error**2).sum(-1)


            #Base Case
            if i == 0:
                prev_cost = cost.mean(-1)
                if track:
                    self.track(R, t,cost.mean().item(), points_2d,points_3d, mask_supported)
            if self.verbose:
                print('Iter ', i, cost.mean().item())
            # print("prev cost is ", prev_cost)


            # Gradient of 3D point P with respect to projection matrix T ([R|t])
            # Nx3x6
            # [[1,0,0,0,-pz,py],
            #  [0,1,0,pz,0,-px],
            #  [0,0,1,-py,px,0]] 
            J_p_T = torch.cat([
                batched_eye_like(points_3d_supported, 3), -skew_symmetric(points_3d_supported)], -1)
            # print("J_p_T is of size ", J_p_T.shape)


            # Gradient of pixel point px with respect to 3D point P
            # Derived from camera projection equation: px = K * [R|t] * P
            # + homogeneous!
            # Nx2x6
            shape = points_3d_supported.shape[:-1]
            o, z = points_3d_supported.new_ones(shape), points_3d_supported.new_zeros(shape)
            J_px_p = torch.stack([
                K[0,0]*o, z, -K[0,0]*points_3d_supported[..., 0] / points_3d_supported[..., 2],
                z, K[1,1]*o, -K[1,1]*points_3d_supported[..., 1] / points_3d_supported[..., 2],
            ], dim=-1).reshape(shape+(2, 3)) / points_3d_supported[..., 2, None, None]

            # feature gradient at projected pixel coordinates
            grad_x_points = indexing_(feature_grad_x, torch.flip(points_2d_supported,(1,)))
            grad_y_points = indexing_(feature_grad_y, torch.flip(points_2d_supported,(1,)))

            # gradient of features with respect to pixel points
            J_f_px = torch.cat((grad_x_points[..., None], grad_y_points[...,None]), -1)

            # gradient of feature error w.r.t. camera matrix T (including K)
            J_e_T = J_f_px @ J_px_p @ J_p_T

            #Grad = J_e_T * error
            Grad = torch.einsum('bij,bi->bj', J_e_T, error)
            Grad = Grad.sum(-2)  # Grad was ... x N x 6

            J = J_e_T # final jacobian
            
            Hess = torch.einsum('ijk,ijl->ikl', J, J) # Approximate Hessian = J.T * J
            Hess = Hess.sum(-3)  # Hess was ... x N x 6 x 6

            # finding gradient step using LM or Newton method
            delta = optimizer_step(Grad, Hess, lambda_, lr = lr)

            # delta = -lr*Grad
            if torch.isnan(delta).any():
                logging.warning('NaN detected, exit')
                break
            
            # Extract deltas (dt = offset translation, dw = rotational offset as vector)
            dt, dw = delta[..., :3], delta[..., 3:6]

            # Rotational delta as a skew symmetric matrix -> SO3
            dr = so3exp_map(dw)
            if self.verbose:
                print("dr is : ", dw)
                print("dt is : ", dt)

            # Update Rotation and translation of camera
            R_new = dr @ R 
            t_new = dr @ t + dt #first rotate old t, then add dt (which is in the new coordinate frame)

            new_points_3d = torch.mm(R_new, pts3D.T).T + t_new
            # Maybe there is some minor problem (+-1 pixel due to rounding!)
            new_points_2d = torch.round(from_homogeneous(torch.mm(K, new_points_3d.T).T)).type(torch.IntTensor) - 1
            
            # Get mask for new points
            layers, height, width  = feature_map_query.shape
            mask_supported = points_within_image(new_points_2d,height, width)

            # Mask new points
            new_points_2d_supported = new_points_2d[mask_supported,:]
            new_points_3d_supported = new_points_3d[mask_supported,:]

            # Check new error
            new_error = indexing_(feature_map_query, torch.flip(new_points_2d_supported, (1,))) - feature_ref[mask_supported]
            new_cost = 0.5*(new_error**2).sum(-1)

            new_cost = new_cost.mean(-1)

            if track:
                    self.track(R_new, t_new,new_cost.item(), new_points_2d,new_points_3d, mask_supported)

            if self.verbose:
                print("new cost is ", new_cost.item())
            lambda_ = np.clip(lambda_ * (10 if new_cost > prev_cost else 1/10),
                              1e-6, 1e4) # according to rule, we change the lambda if error increases/decreases
            
            if new_cost > prev_cost:  # cost increased
                print("cost increased, continue with next iteration")
                # print(lambda_)
                lr = np.clip(0.1*lr, 1e-3, 1.)
                continue
            else:
                lr = lr_reset
            prev_cost = new_cost
            R, t = R_new, t_new # Actually we should write the result only if the cost decreased!
        return R, t