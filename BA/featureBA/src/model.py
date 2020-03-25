from __future__ import print_function, division

from .utils import (from_homogeneous, to_homogeneous,
                batched_eye_like, skew_symmetric, so3exp_map)

from .utils import squared_loss, scaled_loss
import torch
from torch import nn
import numpy as np

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
        @K: Camera matrix
        '''

        if R_init is None: # If R is not inialized, initialize it as identity
            R = torch.eye(3).to(pts3D)
        else:
            R = R_init
        if t_init is None:
            t = pts3D.new_tensor([1, 1, 0]).float()
        else:
            t = t_init

        lambda_ = self.lambda_ # lambda for LM method
        lr = 0.1
        lr_reset = 0.1
        for i in range(self.iterations):

            # project point using current R and T on image
            p_3d_1 = torch.mm(R, pts3D.T).T + t
            p_proj_1 = torch.round(from_homogeneous(torch.mm(K, p_3d_1.T).T)).type(torch.IntTensor)-1
            # print(p_proj_1)
            error = indexing_(feature_map_query, torch.flip(p_proj_1,(1,))) - feature_ref
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
            # print("prev cost is ", prev_cost)

            # Gradient of 3D point P with respect to projection matrix T ([R|t])
            J_p_T = torch.cat([
                batched_eye_like(p_3d_1, 3), -skew_symmetric(p_3d_1)], -1)
            # print("J_p_T is of size ", J_p_T.shape)


            # Gradient of pixel point px with respect to 3D point P
            shape = p_3d_1.shape[:-1]
            o, z = p_3d_1.new_ones(shape), p_3d_1.new_zeros(shape)
            J_px_p = torch.stack([
                K[0,0]*o, z, -K[0,0]*p_3d_1[..., 0] / p_3d_1[..., 2],
                z, K[1,1]*o, -K[1,1]*p_3d_1[..., 1] / p_3d_1[..., 2],
            ], dim=-1).reshape(shape+(2, 3)) / p_3d_1[..., 2, None, None]

            # print("J_px_p is of size ", J_px_p.shape)

            # feature gradient at projected pixel coordinates
            grad_x_points = indexing_(feature_grad_x, torch.flip(p_proj_1,(1,)))
            grad_y_points = indexing_(feature_grad_y, torch.flip(p_proj_1,(1,)))

            # gradient of features with respect to pixel points
            J_f_px = torch.cat((grad_x_points[..., None], grad_y_points[...,None]), -1)

            # print("J_f_px is of size ", J_f_px.shape)
            J_e_T = J_f_px @ J_px_p @ J_p_T # (J_e_T is gradient of error with respect to matrix T)

            Grad = torch.einsum('bij,bi->bj', J_e_T, error)
            # Grad = weights[..., None] * Grad
            Grad = Grad.sum(-2)  # Grad was ... x N x 6

            J = J_e_T # final jacobian
            Hess = torch.einsum('ijk,ijl->ikl', J, J) # Approximate Hessian = J.T * J
            # Hess = weights[..., None, None] * Hess
            Hess = Hess.sum(-3)  # Hess was ... x N x 6 x 6

            # finding gradient step using LM or Newton method
            delta = optimizer_step(Grad, Hess, lambda_, lr = lr)
            # delta = -lr*Grad
            if torch.isnan(delta).any():
                logging.warning('NaN detected, exit')
                break
            dt, dw = delta[..., :3], delta[..., 3:6]
            dr = so3exp_map(dw)
            print("dr is : ", dw)
            print("dt is : ", dt)
            R_new = dr @ R
            t_new = dr @ t + dt

            new_3d_1 = torch.mm(R_new, pts3D.T).T + t_new
            new_proj_1 = torch.round(from_homogeneous(torch.mm(K, new_3d_1.T).T)).type(torch.IntTensor) - 1
            new_error = indexing_(feature_map_query, torch.flip(new_proj_1, (1,))) - feature_ref
            new_cost = 0.5*(new_error**2).sum(-1)
            # new_cost = scaled_loss(new_cost, self.loss_fn, scale[..., None])[0]
            # new_cost = (confidence*new_cost).mean(-1)
            new_cost = new_cost.mean(-1)
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
            R, t = R_new, t_new
            # print(np.linalg.det(R.data.numpy()))
        return R, t