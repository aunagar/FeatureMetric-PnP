from __future__ import print_function, division


import torch
from torch import nn
import numpy as np
import pickle
import kornia
import gin

from helpers.utils import (from_homogeneous, to_homogeneous,
                batched_eye_like, skew_symmetric, so3exp_map)

from helpers.utils import squared_loss, scaled_loss, sobel_filter, cauchy_loss, barron_loss, geman_mcclure_loss


"""
@ToDo:
    - Check if this masking of supported points works as expected (done, seems to work good)
    - We project points 2 times, should decrease to once only (perf.)
    - Function returns the last R and t, which are not necessarily the best (done)
    - Try to explain this spurious oscillations sometimes
    - Try to find a workaround for local minimas, e.g. perturb randomly around best guess
    - Penalize removing points with mask (should add to error)
    - Threshold far-off feature errors (done)
    - If the feature gradient is smaller than the image size, we take the closest one (see
        keypoint_association.py) (done)
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
#    start = torch.cuda.Event(enable_timing=True)
#    end = torch.cuda.Event(enable_timing=True)
    if lambda_:  # LM instead of GN
        D = (H.diagonal(dim1=-2, dim2=-1) + 1e-9).diag_embed()
        H = H + D*lambda_
    try:
#        start.record()
        H_LU, pivots = torch.lu(H)
        # P = torch.inverse(H)
#        end.record()
#        torch.cuda.synchronize()
#        print("Preprocessing time: %lf" % (start.elapsed_time(end)))
    except RuntimeError as e:
        logging.warning(f'Determinant: {torch.det(H)}')
        raise e

#    start.record()
    x = torch.lu_solve(g[..., None], H_LU, pivots)
    delta = -lr * x[..., 0] # generally learning rate does not make sense for second order methods
    # delta = -lr * (P @ g[..., None])[..., 0] # generally learning rate does not make sense for second order methods
#    end.record()
#    torch.cuda.synchronize()
#    print("Solution time: %lf" % (start.elapsed_time(end)))
#    
#    print(H.device)
#    print(H_LU.device)
#    print(delta.device)

    return delta

def indexing_(feature_map, points, width, height):
    '''
    Function gives x and y gradients for 3D points in camera frame.

    inputs: (All pytorch tensors)
    @feature_map : x gradient of the feature map (CxHxW)
    @points : pixel coordinates of points (Nx2)

    @imsize: optional: tuple(height, width)

    outputs: 
    features : features for the points (NxC) 
    '''

    points_row = (points[:,0].double() * feature_map.shape[-2] / height).floor().type(torch.ShortTensor) #Should be short
    points_col = (points[:,1].double() * feature_map.shape[-1] / width).floor().type(torch.ShortTensor)

    features = []
    for i,j in zip(points_row, points_col):
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


def ratio_threshold_feature_errors(feature_errors, threshold = 0.8):
    """
    Function returns a mask (True,False) with feature errors that are < max(feature_errors) * threshold
    """
    limit = torch.max(torch.abs(feature_errors)) * threshold
    mask = (torch.abs(feature_errors) < limit)

    # TODO: Minimum inliers as a parameter (Measure distribution)

    return mask

@gin.configurable
def find_inliers(pts3D, R, t, feature_map_query, feature_ref, K,im_width, im_height, threshold=None, loss_fn = squared_loss, mode = "ratio_max"):
    # project all points using current R and T on image 
    points_3d = torch.mm(R, pts3D.T).T + t
    points_2d = torch.round(from_homogeneous(torch.mm(K, points_3d.T).T)).type(torch.IntTensor)-1

    # Get the points that are supported within our image size
    mask_supported = points_within_image(points_2d, im_width, im_height)

    # We only take supported points
    points_2d_supported = points_2d[mask_supported,:]
    points_3d_supported = points_3d[mask_supported,:]

    error = indexing_(feature_map_query, torch.flip(points_2d_supported,(1,)), im_width, im_height) - feature_ref[mask_supported]
    
    cost = 0.5 * (error**2).sum(-1) # Why 0.5??
    cost_full, weights, _ = loss_fn(cost)

    if mode == "ratio_max":
        threshold_mask = ratio_threshold_feature_errors(cost_full, threshold = threshold) #Mask!
        mask_supported[mask_supported] = threshold_mask
        return mask_supported


@gin.configurable
class sparseFeaturePnP(nn.Module):
    def __init__(self, n_iters, loss_fn = squared_loss, lambda_ = 0.01,
                verbose = False, ratio_threshold = None, useGPU = False):
        super().__init__()
        self.iterations = n_iters
        self.loss_fn = loss_fn
        self.verbose = verbose
        self.lambda_ = lambda_
        self.track_ = {"Rs":[], "ts":[], "costs": [], "points2d":[], "mask":[], "threshold_mask":[]}
        self.use_ratio_test_ = ratio_threshold is not None
        self.ratio_threshold_ = ratio_threshold
        self.initial_cost_=None
        self.useGPU = useGPU # GPU works but might still need some modifications

    def track(self, R,t, cost, points_2d,  mask, threshold_mask):
        self.track_["Rs"].append(R)
        self.track_["ts"].append(t)
        self.track_["costs"].append(cost)
        self.track_["points2d"].append(points_2d)
        self.track_["mask"].append(mask)
        self.track_["threshold_mask"].append(threshold_mask)

    @gin.configurable
    def multilevel_optimization(self, feature_pyramid, pts3D, feature_ref, feature_map_query,
                    feature_grad_x, feature_grad_y,*args, **kwargs):
                    
        self.initial_cost_= self.compute_cost(pts3D, kwargs["R_init"], kwargs["t_init"], feature_map_query, feature_ref, *args)
        channels = feature_map_query.shape[0]

        if feature_pyramid is None:
            return self.forward(*args,**kwargs)
        else:
            for start,end, target_size, kernel_size in feature_pyramid:
                feature_map_local = feature_map_query[start:end,:,:].unsqueeze(0)
                
                if target_size is not None:
                    feature_map_local = nn.functional.interpolate(feature_map_local, size=(target_size, target_size),mode='bilinear')
                if kernel_size is not None:
                    gauss = kornia.filters.get_gaussian_kernel2d((kernel_size, kernel_size), (1.0, 1.0)).unsqueeze(0).unsqueeze(0).repeat(end-start,1,1,1)
                    feature_map_local = nn.functional.conv2d(feature_map_local, weight=gauss.double(), groups=end-start, padding=1)

                feature_map_local = feature_map_local.squeeze(0)

                if target_size is None and kernel_size is None:
                    grad_x, grad_y = feature_grad_x[start:end,:,:], feature_grad_y[start:end,:,:]
                else:
                    grad_x, grad_y = sobel_filter(feature_map_local)

                kwargs["R_init"], kwargs["t_init"] = self.forward(pts3D, feature_ref[:,start:end], feature_map_local,
                    grad_x, grad_y,*args,**kwargs)
                del feature_map_local, grad_x, grad_y
                # print("Cost:", self.compute_cost(pts3D, kwargs["R_init"], kwargs["t_init"], feature_map_query, feature_ref, *args).item())
        return kwargs["R_init"], kwargs["t_init"]


    def compute_cost(self, pts3D, R, t, feature_map_query, feature_ref, K,im_width, im_height):
        points_3d = torch.mm(R, pts3D.T).T + t
        points_2d = torch.round(from_homogeneous(torch.mm(K, points_3d.T).T)).type(torch.IntTensor)-1

        # Get the points that are supported within our image size
        mask_supported = points_within_image(points_2d,im_width, im_height)

        # We only take supported points
        points_2d_supported = points_2d[mask_supported,:]
        points_3d_supported = points_3d[mask_supported,:]

        error = indexing_(feature_map_query, torch.flip(points_2d_supported,(1,)), im_width, im_height) - feature_ref[mask_supported]

        if self.use_ratio_test_:
            cost_full = 0.5*(error**2).sum(-1)
            threshold_mask = ratio_threshold_feature_errors(cost_full, threshold = self.ratio_threshold_)
            error = error[threshold_mask,:]
            points_2d_supported = points_2d_supported[threshold_mask,:]
            points_3d_supported = points_3d_supported[threshold_mask,:]
            #mask_supported[mask_supported] = threshold_mask
            #outliers = threshold_mask.nonzero()
        else:
            threshold_mask = None
        # Cost of Feature error -> SSD
        cost = 0.5*(error**2).sum(-1)
        return cost.mean()

    @gin.configurable
    def forward(self, pts3D, feature_ref, feature_map_query,
                feature_grad_x, feature_grad_y, K, im_width, im_height, R_init=None, t_init=None,
                track = False, confidence=None, scale=None):
        '''
        inputs:
        @pts3D : 3D points in reference camera frame (Nx3)
        @feature_ref : features for these 3D points in reference frame (NxC)
        @feature_map_query : feature map for the query image (CxHxW)
        @feature_grad_x : feature gradient map for query image (CxHxW)
        @feature_grad_y : feature gradient map for query image (CxHxW)
        @K : Camera matrix

        optional:
        @R_init : Initial rotation matrix (3x3)
        @t_init : Initial translation vector (3x1)
        @track : Whether a history of parameters should be written to self.track_
        '''

        # if not track:
        #     track = track_
        #     pickle_path = pickle_path_
        # else:
        #     pickle_path = None

        # This might be convenient but might to lead to errors
        if R_init is None: # If R is not initalized, initialize it as identity
            R = torch.eye(3).to(pts3D)
        else:
            R = R_init
        if t_init is None:
            t = pts3D.new_tensor([1, 1, 0]).double()
        else:
            t = t_init

        # regularization parameter
        lambda_ = self.lambda_ # lambda for LM method
        lr = 1.0 # learning rate
        lr_reset = 1.0 #reset learning rate

        #no_outliers = torch.new_empty((0,1))

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        # end.record()
        # torch.cuda.synchronize()
        # print("iter %d: %f" % (i, start.elapsed_time(end)))

        if self.useGPU and torch.cuda.is_available(): # Move stuff to GPU
            # print("Running Sparse3DBA.forward on GPU")
            K, R, t, pts3D = K.cuda(), R.cuda(), t.cuda(), pts3D.cuda()
            feature_map_query, feature_ref = feature_map_query.cuda(), feature_ref.cuda()
            feature_grad_x, feature_grad_y = feature_grad_x.cuda(), feature_grad_y.cuda()

        for i in range(self.iterations):

            # project all points using current R and T on image 
            points_3d = torch.mm(R, pts3D.T).T + t

            if self.useGPU:                
                points_2d = torch.round(from_homogeneous(torch.mm(K, points_3d.T).T)).type(torch.cuda.IntTensor)-1
            else:
                points_2d = torch.round(from_homogeneous(torch.mm(K, points_3d.T).T)).type(torch.IntTensor)-1

            # Get the points that are supported within our image size
            mask_supported = points_within_image(points_2d, im_width, im_height)

            # We only take supported points
            points_2d_supported = points_2d[mask_supported,:]
            points_3d_supported = points_3d[mask_supported,:]

            error = indexing_(feature_map_query, torch.flip(points_2d_supported,(1,)), im_width, im_height) - feature_ref[mask_supported]
            
            


            if self.use_ratio_test_:
                cost = 0.5 * (error**2).sum(-1) # Why 0.5??
                cost_full, weights, _ = self.loss_fn(cost)
                #cost_full, weights, _ = scaled_loss(
                #    cost_full, self.loss_fn, scale[..., None])
                threshold_mask = ratio_threshold_feature_errors(cost_full, threshold = self.ratio_threshold_)
                error = error[threshold_mask,:]
                points_2d_supported = points_2d_supported[threshold_mask,:]
                points_3d_supported = points_3d_supported[threshold_mask,:]
                # mask_supported[mask_supported] = threshold_mask
                # outliers = threshold_mask.nonzero()
            else:
                threshold_mask = None
            # Cost of Feature error -> SSD
            cost = 0.5 * (error**2).sum(-1)
            cost, weights, _ = self.loss_fn(cost)


            # if confidence is not None:
            #     weights = weights * confidence
            #     cost = cost * confidence

            # Base Case
            if i == 0:
                prev_cost = cost.mean(-1)
                self.best_cost_ = prev_cost
                num_inliers = points_2d_supported.shape[0]
                R_best = R
                t_best = t

                self.best_num_inliers_  = num_inliers
                if self.initial_cost_ is None:
                    self.initial_cost_ = prev_cost

                if track:
                    self.track(R, t,cost.mean().item(), points_2d, mask_supported, threshold_mask)
            if self.verbose:
                print('Iter ', i, cost.mean().item())


            # Gradient of 3D point P with respect to projection matrix T ([R|t])
            # Nx3x6
            # [[1,0,0,0,-pz,py],
            #  [0,1,0,pz,0,-px],
            #  [0,0,1,-py,px,0]] 
            J_p_T = torch.cat([
                batched_eye_like(points_3d_supported, 3), -skew_symmetric(points_3d_supported)], -1)
            # print("J_p_T is  ", J_p_T)

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

            # print("J_px_p is ", J_px_p)
            # feature gradient at projected pixel coordinates
            grad_x_points = indexing_(feature_grad_x, torch.flip(points_2d_supported,(1,)), im_width, im_height)
            grad_y_points = indexing_(feature_grad_y, torch.flip(points_2d_supported,(1,)), im_width, im_height)

            # gradient of features with respect to pixel points
            J_f_px = torch.cat((grad_x_points[..., None], grad_y_points[...,None]), -1)

            # print("J_f_px is ", J_f_px)
            # gradient of feature error w.r.t. camera matrix T (including K)
            J_e_T = J_f_px @ J_px_p @ J_p_T
            
            #Grad = J_e_T * error
            Grad = torch.einsum('bij,bi->bj', J_e_T, error)
            Grad = weights[..., None] * Grad # Weights are the first derivative of the loss function
            Grad = Grad.sum(-2)  # Grad was ... x N x 6

            J = J_e_T # final jacobian
            
            Hess = torch.einsum('ijk,ijl->ikl', J, J) # Approximate Hessian = J.T * J
            Hess = weights[..., None, None] * Hess
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
            # if self.verbose:
                # print("dr is : ", dw)
                # print("dt is : ", dt)

            # Update Rotation and translation of camera
            R_new = dr @ R 
            t_new = dr @ t + dt #first rotate old t, then add dt (which is in the new coordinate frame)

            new_points_3d = torch.mm(R_new, pts3D.T).T + t_new
            # Maybe there is some minor problem (+-1 pixel due to rounding!)
            if self.useGPU:                
                new_points_2d = torch.round(from_homogeneous(torch.mm(K, new_points_3d.T).T)).type(torch.cuda.IntTensor) - 1
            else:
                new_points_2d = torch.round(from_homogeneous(torch.mm(K, new_points_3d.T).T)).type(torch.IntTensor) - 1
            
            # Get mask for new points
            mask_supported = points_within_image(new_points_2d, im_width, im_height)

            # Mask new points
            new_points_2d_supported = new_points_2d[mask_supported,:]
            new_points_3d_supported = new_points_3d[mask_supported,:]
            
            # Check new error
            new_error = indexing_(feature_map_query, torch.flip(new_points_2d_supported, (1,)), im_width, im_height) - feature_ref[mask_supported]
            
            if self.use_ratio_test_:
                new_cost = 0.5 * (new_error**2).sum(-1)
                new_cost_full, weights, _ = self.loss_fn(new_cost)

                new_threshold_mask = ratio_threshold_feature_errors(new_cost_full, threshold = self.ratio_threshold_)
                new_error = new_error[new_threshold_mask]
                new_points_2d_supported = new_points_2d_supported[new_threshold_mask,:]
                new_points_3d_supported = new_points_3d_supported[new_threshold_mask,:]
                #mask_supported[mask_supported] = new_threshold_mask
            else:
                new_threshold_mask = None
            new_cost = 0.5 * (new_error**2).sum(-1)
            new_cost, weights, _ = self.loss_fn(new_cost)
            new_cost = new_cost.mean()

            if track:
                    self.track(R_new, t_new,new_cost.item(), new_points_2d, mask_supported, new_threshold_mask)
            
            if self.verbose:
                print("new cost is ", new_cost.item())
            lambda_ = np.clip(lambda_ * (10 if new_cost > prev_cost else 1/10),
                              1e-6, 1e4) # according to rule, we change the lambda if error increases/decreases
            
            if new_cost > prev_cost:  # cost increased
                #print("cost increased, continue with next iteration")
                # print(lambda_)
                lr = np.clip(0.1*lr, 1e-3, 1.)
                continue
            else:
                lr = lr_reset
                if new_cost < self.best_cost_:
                    R_best = R_new
                    t_best = t_new
                    self.best_num_inliers_ = new_points_2d_supported.shape[0]
                    self.best_cost_ = new_cost
            prev_cost = new_cost
            
            R, t = R_new, t_new # Actually we should write the result only if the cost decreased!

        # if track and (pickle_path is not None):
        #     pickle.dump(self.track_, open(pickle_path, "wb")) 

        if self.useGPU:
            return R_best.cpu(), t_best.cpu()
        else:
            return R_best, t_best
