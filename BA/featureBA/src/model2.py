from __future__ import print_function, division

from .utils import (from_homogeneous, to_homogeneous,
                batched_eye_like, skew_symmetric, so3exp_map)
import cv2
from .utils import squared_loss, scaled_loss
import torch
from torch import nn
import numpy as np

def optimizer_step(g, H, lambda_=0, lr = 0.001, optimizer = 'newton'):
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
    if optimizer == 'gd':
        delta = -lr * g
    else:
        delta = -(P @ g[..., None])[..., 0]
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

    features = torch.cat([feature_map[:, i, j].unsqueeze(0) for i, j in zip(points[:,0], points[:,1])])

    return features


class sparse3DBA(nn.Module):
    def __init__(self, n_iters, loss_fn = squared_loss, lambda_ = 0.01,
                opt_depth=True, verbose=False):
        super().__init__()
        self.iterations = n_iters
        self.loss_fn = loss_fn
        self.verbose = verbose
        self.lambda_ = lambda_

    def forward(self, pts3D, feature_ref, feature_map_query,
                feature_grad_x, feature_grad_y, K, R_init=None, t_init=None,
                confidence=None, scale=None):
        '''
        inputs:
        @pts3D : 3D points in reference camera frame (Nx3)
        @feature_ref: features for these 3D points in reference frame (NxC)
        @feature_map_query: feature map for the query image (CxHxW)
        @feature_grad_x : feature gradient map for query image (CxHxW)
        @feature_grad_y : feature gradient map for query image (CxHxW)
        '''
        if R_init is None:
            R = torch.eye(3).to(pts3D)
        else:
            R = R_init
        if t_init is None:
            t = pts3D.new_tensor([1, 1, 0]).float()
        else:
            t = t_init

        lambda_ = self.lambda_
        for i in range(self.iterations):
            p_3d_1 = torch.mm(R.T, pts3D.T).T + t
            p_proj_1 = from_homogeneous(torch.mm(K, p_3d_1.T).T).type(torch.IntTensor)-1
            
            error = torch.flip(p_proj_1, (1,)) - feature_ref
            error = error.type(torch.DoubleTensor)
            # error = indexing_(feature_map_query, torch.flip(p_proj_1,(1,))) - feature_ref
            cost = 0.5*(error**2).sum(-1)

            # cost, weights, _ = scaled_loss(
            #     cost, self.loss_fn, scale[..., None])
            # if confidence is not None:
            #     weights = weights * confidence
            #     cost = cost * confidence
            if i == 0:
                prev_cost = cost.mean(-1)
            if self.verbose:
                print('Iter ', i, cost.mean().item())
            print("prev cost is ", prev_cost)
            J_p_T = torch.cat([
                batched_eye_like(p_3d_1, 3), -skew_symmetric(p_3d_1)], -1)
            print("J_p_T is ", J_p_T.shape)
            shape = p_3d_1.shape[:-1]
            o, z = p_3d_1.new_ones(shape), p_3d_1.new_zeros(shape)

            J_e_p = torch.stack([
                K[0,0]*o, z, -K[0,0]*p_3d_1[..., 0] / p_3d_1[..., 2],
                z, K[1,1]*o, -K[1,1]*p_3d_1[..., 1] / p_3d_1[..., 2],
            ], dim=-1).reshape(shape+(2, 3)) / p_3d_1[..., 2, None, None]

            # J_e_p = 
            print("J_e_p is ", J_e_p.shape)

            grad_x_points = indexing_(feature_grad_x, torch.flip(p_proj_1,(1,)))
            grad_y_points = indexing_(feature_grad_y, torch.flip(p_proj_1,(1,)))

            J_p_F = torch.cat((grad_x_points[..., None], grad_y_points[...,None]), -1)

            # print("J_p_F is ", J_p_F.shape)
            # J_e_T = J_p_F @ J_e_p @ J_p_T

            J_e_T = J_e_p @ J_p_T

            Grad = torch.einsum('...ijk,...ij->...ik', J_e_T, error)
            # Grad = weights[..., None] * Grad
            Grad = Grad.sum(-2)  # Grad was ... x N x 6

            J = J_e_T
            Hess = torch.einsum('...ijk,...ijl->...ikl', J, J)
            # Hess = weights[..., None, None] * Hess
            Hess = Hess.sum(-3)  # Hess was ... x N x 6 x 6

            delta = optimizer_step(Grad, Hess, lambda_)
            # delta = -1e-5*Grad
            if torch.isnan(delta).any():
                logging.warning('NaN detected, exit')
                break
            dt, dw = delta[..., :3], delta[..., 3:6]
            dr = so3exp_map(dw)
            print("dr is : ", dw)
            print("dt is : ", dt)
            R_new = dr @ R
            t_new = dr @ t + dt

            new_proj_1 = from_homogeneous(torch.mm(K, (pts3D @ R_new.T + t_new).T).T).type(torch.IntTensor) - 1
            # print(new_proj_1)
            new_error = torch.flip(new_proj_1, (1,)) - feature_ref
            new_error = new_error.type(torch.DoubleTensor)
            # new_error = indexing_(feature_map_query, torch.flip(new_proj_1, (1,))) - feature_ref
            new_cost = 0.5*(new_error**2).sum(-1)
            # new_cost = scaled_loss(new_cost, self.loss_fn, scale[..., None])[0]
            # new_cost = (confidence*new_cost).mean(-1)
            new_cost = new_cost.mean(-1)
            print("new cost is ", new_cost)
            lambda_ = np.clip(lambda_ * (10 if new_cost > prev_cost else 1/10),
                              1e-6, 1e4)
            if new_cost > prev_cost:  # cost increased
                print("cost increased, continue with next iteration")
                print(lambda_)
                continue
            prev_cost = new_cost
            R, t = R_new, t_new
            # print(np.linalg.det(R.data.numpy()))
        return R, t