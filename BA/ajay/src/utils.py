"""
A set of geometry tools for PyTorch tensors and sometimes NumPy arrays.
"""

import torch
import numpy as np
import kornia

def sobel_filter(feature_map, batch = False):
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
        grad = kornia.filters.SpatialGradient(feature_map)
        return grad[:,:,0,:,:]
    else:
        feature_map = feature_map[None,...]
        grad = kornia.filters.SpatialGradient(feature_map)
        return grad[:,:,0,:,:].reshape(feature_map.shape), grad[:,:,1,:,:].reshape(feature_map.shape)

def to_homogeneous(points):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1]+(1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1]+(1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError


def from_homogeneous(points):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / points[..., -1:]


def batched_eye_like(x, n):
    """Create a batch of identity matrices.
    Args:
        x: a reference torch.Tensor whose batch dimension will be copied.
        n: the size of each identity matrix.
    Returns:
        A torch.Tensor of size (B, n, n), with same dtype and device as x.
    """
    return torch.eye(n).to(x)[None].repeat(len(x), 1, 1)


def create_norm_matrix(shift, scale):
    """Create a normalization matrix that shifts and scales points."""
    T = batched_eye_like(shift, 3)
    T[:, 0, 0] = T[:, 1, 1] = scale
    T[:, :2, 2] = shift
    return T


def normalize_keypoints(kpts, size=None, shape=None):
    """Normalize a set of 2D keypoints for input to a neural network.

    Perform the normalization according to the size of the corresponding
    image: shift by half and scales by the longest edge.
    Use either the image size or its tensor shape.

    Args:
        kpts: a batch of N D-dimensional keypoints: (B, N, D).
        size: a tensor of the size the image `[W, H]`.
        shape: a tuple of the image tensor shape `(B, C, H, W)`.
    """
    if size is None:
        assert shape is not None
        _, _, h, w = shape
        one = kpts.new_tensor(1)
        size = torch.stack([one*w, one*h])[None]

    shift = size.float() / 2
    scale = size.max(1).values.float() / 2  # actual SuperGlue mult by 0.7
    kpts = (kpts - shift[:, None]) / scale[:, None, None]

    T_norm = create_norm_matrix(shift, scale)
    T_norm_inv = create_norm_matrix(-shift/scale[:, None], 1./scale)
    return kpts, T_norm, T_norm_inv


def skew_symmetric(v):
    """Create a skew-symmetric matrix from a (batched) vector of size (..., 3).
    """
    z = torch.zeros_like(v[..., 0])
    M = torch.stack([
        z, -v[..., 2], v[..., 1],
        v[..., 2], z, -v[..., 0],
        -v[..., 1], v[..., 0], z,
    ], dim=-1).reshape(v.shape[:-1]+(3, 3))
    return M


def T_to_E(T):
    """Convert batched poses (..., 4, 4) to batched essential matrices."""
    return T[..., :3, :3] @ skew_symmetric(T[..., :3, 3])


def sym_epipolar_distance(p0, p1, E):
    """Compute batched symmetric epipolar distances.
    Args:
        p0, p1: batched tensors of N 2D points of size (..., N, 2).
        E: essential matrices from camera 0 to camera 1, size (..., 3, 3).
    Returns:
        The symmetric epipolar distance of each point-pair: (..., N).
    """
    if p0.shape[-1] != 3:
        p0 = to_homogeneous(p0)
    if p1.shape[-1] != 3:
        p1 = to_homogeneous(p1)
    p1_E_p0 = torch.einsum('...ni,...ij,...nj->...n', p1, E, p0)
    E_p0 = torch.einsum('...ij,...nj->...ni', E, p0)
    Et_p1 = torch.einsum('...ij,...ni->...nj', E, p1)
    d = p1_E_p0**2 * (
        1. / (E_p0[..., 0]**2 + E_p0[..., 1]**2 + 1e-15) +
        1. / (Et_p1[..., 0]**2 + Et_p1[..., 1]**2 + 1e-15))
    return d


def so3exp_map(w):
    """Compute rotation matrices from batched twists.
    Args:
        w: batched 3D axis-angle vectors of size (..., 3).
    Returns:
        A batch of rotation matrices of size (..., 3, 3).
    """
    theta = w.norm(p=2, dim=-1, keepdim=True)
    W = skew_symmetric(w / theta)
    theta = theta[..., None]  # ... x 1 x 1
    res = W * torch.sin(theta) + (W @ W) * (1 - torch.cos(theta))
    res = torch.where(theta < 1e-12, torch.zeros_like(res), res)
    return torch.eye(3).to(W) + res
