import torch
from torch import nn
import logging
import numpy as np
import kornia

from helpers.utils import (from_homogeneous, to_homogeneous,
                batched_eye_like, skew_symmetric, so3exp_map)

from helpers.utils import squared_loss, scaled_loss, sobel_filter, cauchy_loss, barron_loss, geman_mcclure_loss


def optimizer_step(g, H, lambda_=0, lr=1.):
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
        H_LU, pivots = torch.lu(H)
    except RuntimeError as e:
        logging.warning(f'Determinant: {torch.det(H)}')
        raise e
    x = torch.lu_solve(g[..., None], H_LU, pivots)
    delta = -lr * x[..., 0]
    return delta


def mask_in_image(pts, image_size, pad=1):
    w, h = image_size
    return torch.all((pts >= pad) & (pts < pts.new_tensor([w-pad, h-pad])), -1)


def interpolate_tensor(tensor, pts):
    '''
    Assume that the points are already scales to the size of the tensor
    tensor: C x H x W
    pts: N x 2
    return N x C
    '''
    c, h, w = tensor.shape
    pts = (pts / pts.new_tensor([w-1, h-1])) * 2 - 1
    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    tensor = torch.nn.functional.grid_sample(
        tensor[None], pts[None, None], mode='bilinear', **args)
    return tensor.reshape(c, -1).t()


def feature_gradient(feature_map, batch=False):
    """
    Apply 3x3 sobel filter on the n-dimensional feature map

    inputs:
    @feature_map : BxCxHxW (B is required if batch = True)
    @batch : is it batched or not

    outputs:
    grad_x (BxCxHxW)
    grad_y (BxCxHxW)
    """
    # we need 3D feature map
    assert len(feature_map.shape) > 2

    if batch:
        grad = kornia.filters.spatial_gradient(feature_map)
        return grad[:, :, 0, :, :], grad[:, :, 1, :, :]
    else:
        feature_map = feature_map[None]
        grad = kornia.filters.spatial_gradient(feature_map)
        grad_x = grad[:, :, 0, :, :].reshape(feature_map.shape[1:])
        grad_y = grad[:, :, 1, :, :].reshape(feature_map.shape[1:])
        return grad_x, grad_y


def radial_distortion(p, k):
    # [x_norm, y_norm] = [x, y] * (1 + kr^2)
    r = torch.sum(p**2, -1, keepdim=True)
    return p * (1 + r * k)


class S2DPnP(nn.Module):
    def __init__(self, iterations, loss_fn=squared_loss, lambda_=0.01,
                 scaling=1.0, verbose=False, normalize_feat=True,
                 pad=1, do_GN=False):
        super().__init__()
        self.iterations = iterations
        self.loss_fn = loss_fn
        self.verbose = verbose
        self.lambda_ = lambda_
        self.scaling = scaling
        self.normalize_feat = normalize_feat
        self.pad = pad
        self.do_GN = do_GN

    def forward(self, p3D, feature_ref, feature_query, K, R_init, t_init,
                rdist=None):
        grad_x, grad_y = feature_gradient(feature_query)
        image_size = feature_query.shape[-2:][::-1]
        R, t = R_init, t_init
        lambda_ = self.lambda_
        lr = lr_reset = self.scaling

        if self.normalize_feat:
            feature_ref = torch.nn.functional.normalize(feature_ref, dim=1)

        for i in range(self.iterations):
            p3D_q = (p3D @ R.T) + t
            p2D_norm_d = from_homogeneous(p3D_q)
            if rdist is not None:
                p2D_norm = radial_distortion(p2D_norm_d, rdist)
            else:
                p2D_norm = p2D_norm_d
            p2D = p2D_norm * K[..., [0, 1], [0, 1]] + K[..., [0, 1], [2, 2]]

            if i == 0:
                p2D_init = p2D.detach()

            valid = mask_in_image(p2D, image_size, pad=self.pad)
            assert torch.any(valid), 'No valid projection, reduce padding?'
            p3D_q = p3D_q[valid]
            p2D_norm_d = p2D_norm_d[valid]
            p2D_norm = p2D_norm[valid]
            p2D = p2D[valid]

            feature_p2D_raw = interpolate_tensor(feature_query, p2D)
            if self.normalize_feat:
                feature_p2D = torch.nn.functional.normalize(
                    feature_p2D_raw, dim=1)
            else:
                feature_p2D = feature_p2D_raw

            error = feature_p2D - feature_ref[valid]
            cost = (error**2).sum(-1)
            cost, weights, _ = self.loss_fn(cost)

            if i == 0:
                cost_best = cost.mean(-1)
                if self.verbose:
                    logging.info(f'Initial cost: {cost_best.item():.4E}')

            # Gradient of 3D point P with respect to projection matrix ([R|t])
            # N x 3 x 6
            # [[1,0,0,0,-pz,py],
            #  [0,1,0,pz,0,-px],
            #  [0,0,1,-py,px,0]]
            J_p3_T = torch.cat([
                batched_eye_like(p3D_q, 3), -skew_symmetric(p3D_q)], -1)

            # Gradient of normalized point pn with respect to 3D point p3
            # N x 2 x 3
            x, y, d = p3D_q[..., 0], p3D_q[..., 1], p3D_q[..., 2]
            z = torch.zeros_like(d)
            J_pn_p3 = torch.stack([
                1/d, z, -x / d**2,
                z, 1/d, -y / d**2], dim=-1).reshape(p3D_q.shape[:-1]+(2, 3))

            if rdist is not None:
                # Gradient of radial distortion
                # N x 2 x 2 matrix
                # [[1+2kx 2ky]
                #  [2kx   1+2ky]]
                J_dist = (batched_eye_like(p2D_norm, 2)
                          + 2 * rdist * p2D_norm_d[..., None, :])
                J_pn_p3 = J_dist @ J_pn_p3
                del J_dist

            # Gradient of intrinsic matrix K
            z = torch.zeros_like(K[..., 0, 0])
            J_p2_pn = torch.stack([
                K[..., 0, 0], z,
                z, K[..., 1, 1]], dim=-1).view(K.shape[:-2]+(1, 2, 2))

            # Gradient of features
            # N x D x 2
            grad_x_p2D = interpolate_tensor(grad_x, p2D)
            grad_y_p2D = interpolate_tensor(grad_y, p2D)
            J_f_p2 = torch.stack([grad_x_p2D, grad_y_p2D], -1)
            del grad_x_p2D, grad_y_p2D

            if self.normalize_feat:
                # Gradient of L2 normalization
                norm = torch.norm(feature_p2D_raw, p=2, dim=1)
                normed = feature_p2D
                Id = torch.eye(normed.shape[-1])[None].to(normed)
                J_norm_f = (Id - normed[:, :, None] @ normed[:, None])
                J_norm_f = J_norm_f / norm[..., None, None]
                J_f_p2 = J_norm_f @ J_f_p2
                del J_norm_f

            # N x D x 6
            J = J_f_p2 @ J_p2_pn @ J_pn_p3 @ J_p3_T
            del J_f_p2, J_p2_pn, J_pn_p3, J_p3_T

            # jacobi scaling
            if i == 0:
                jacobi = 1 / (1 + torch.norm(J, p=2, dim=(0, 1)))
            J = J * jacobi[None, None]

            Grad = torch.einsum('...ndi,...nd->...ni', J, error)
            Grad = weights[..., None] * Grad
            Grad = Grad.sum(-2)  # Grad was ... x N x 6

            Hess = torch.einsum('...ijk,...ijl->...ikl', J, J)
            Hess = weights[..., None, None] * Hess
            Hess = Hess.sum(-3)  # Hess was ... x N x 6 x 6

            delta = optimizer_step(
                Grad, Hess, 0 if self.do_GN else lambda_, lr=lr)
            del Grad, Hess
            if torch.isnan(delta).any():
                logging.warning('NaN detected, exit')
                break
            delta = delta * jacobi

            dt, dw = delta[..., :3], delta[..., 3:6]
            dr = so3exp_map(dw)
            R_new = dr @ R
            t_new = dr @ t + dt

            p2D_new = from_homogeneous((p3D @ R_new.T) + t_new)
            if rdist is not None:
                p2D_new = radial_distortion(p2D_new, rdist)
            p2D_new = p2D_new * K[..., [0, 1], [0, 1]] + K[..., [0, 1], [2, 2]]
            valid_new = mask_in_image(p2D_new, image_size, pad=self.pad)
            feature_p2D_new = interpolate_tensor(feature_query, p2D_new)
            if self.normalize_feat:
                feature_p2D_new = nn.functional.normalize(
                        feature_p2D_new, dim=1)
            error_new = feature_p2D_new[valid_new] - feature_ref[valid_new]
            cost_new = self.loss_fn((error_new**2).sum(-1))[0].mean(-1)
            if self.verbose:
                logging.info(
                    f'it {i} New cost: {cost_new.item():.4E} {lambda_:.1E}')

            if not self.do_GN:
                lambda_ = np.clip(
                    lambda_ * (10 if cost_new > cost_best else 1/10),
                    1e-8, 1e4)

            if cost_new > cost_best:  # cost increased
                # lr = np.clip(lr / 10, 1e-3, 1.)
                continue

            lr = lr_reset
            cost_best = cost_new
            R, t = R_new, t_new

        p2d_final = from_homogeneous((p3D.detach() @ R.T) + t)
        if rdist is not None:
            p2d_final = radial_distortion(p2d_final, rdist)
        p2d_final = p2d_final*K[..., [0, 1], [0, 1]]+K[..., [0, 1], [2, 2]]
        feature_p2D_i = interpolate_tensor(feature_query, p2D_init)
        feature_p2D_f = interpolate_tensor(feature_query, p2d_final)
        if self.normalize_feat:
            feature_p2D_i = nn.functional.normalize(feature_p2D_i, dim=1)
            feature_p2D_f = nn.functional.normalize(feature_p2D_f, dim=1)
        cost_init = ((feature_ref - feature_p2D_i)**2).sum(-1)
        cost_final = ((feature_ref - feature_p2D_f)**2).sum(-1)
        NA = cost_init.new_tensor(float('nan'))
        cost_init = torch.where(
            mask_in_image(p2D_init, image_size, pad=self.pad), cost_init, NA)
        cost_final = torch.where(
            mask_in_image(p2d_final, image_size, pad=self.pad), cost_final, NA)

        if self.verbose:
            # diff_R, diff_t = pose_error(R_init, t_init, R, t)
            # logging.info(f'Change in R, t: {diff_R:.2E} deg, {diff_t:.2E} m')
            diff_p2D = torch.norm(p2D_init - p2d_final, p=2, dim=-1)
            logging.info(
                f'Change in points: mean {diff_p2D.mean().item():.2E}, '
                f'max {diff_p2D.max().item():.2E}')
            valid = (~torch.isnan(cost_init)) & (~torch.isnan(cost_final))
            better = torch.mean((cost_init > cost_final).float()[valid])
            logging.info(f'Improvement for {better*100:.3f}% of points')

        return R, t, cost_init, cost_final
