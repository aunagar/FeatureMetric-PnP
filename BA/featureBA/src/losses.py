"""
Generic losses and error functions for optimization or training deep networks.
"""

import torch
import math

from .utils import to_homogeneous, from_homogeneous


def scaled_loss(x, fn, a):
    """Apply a loss function to a tensor and pre- and post-scale it.
    Args:
        x: the data tensor, should already be squared: `x = y**2`.
        fn: the loss function, with signature `fn(x) -> y`.
        a: the scale parameter.
    Returns:
        The value of the loss, and its first and second derivatives.
    """
    a2 = a**2
    loss, loss_d1, loss_d2 = fn(x/a2)
    return loss*a2, loss_d1, loss_d2/a2


def squared_loss(x):
    """A dummy squared loss."""
    return x, torch.ones_like(x), torch.zeros_like(x)


def huber_loss(x):
    """The classical robust Huber loss, with first and second derivatives."""
    mask = x <= 1
    sx = torch.sqrt(x)
    isx = torch.max(sx.new_tensor(torch.finfo(torch.float).eps), 1/sx)
    loss = torch.where(mask, x, 2*sx-1)
    loss_d1 = torch.where(mask, torch.ones_like(x), isx)
    loss_d2 = torch.where(mask, torch.zeros_like(x), -isx/(2*x))
    return loss, loss_d1, loss_d2


def barron_loss(x, alpha):
    """Parameterized  & adaptive robust loss function.
    Described in:
        A General and Adaptive Robust Loss Function, Barron, CVPR 2019

    Contrary to the original implementation, assume the the input is already
    squared and scaled (basically scale=1). Computes the first derivative, but
    not the second (TODO if needed).
    """
    loss_two = x
    loss_two_d1 = torch.ones_like(x)

    loss_zero = 2 * torch.log1p(torch.min(0.5*x, x.new_tensor(33e37)))
    loss_zero_d1 = 2 / (x + 2)

    # The loss when not in one of the above special cases.
    machine_epsilon = torch.tensor(torch.finfo(torch.float32).eps).to(x)
    # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
    beta_safe = torch.max(machine_epsilon, torch.abs(alpha - 2.))
    # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
    alpha_safe = torch.where(alpha >= 0, torch.ones_like(alpha),
                             -torch.ones_like(alpha)) * torch.max(
                                 machine_epsilon, torch.abs(alpha))

    loss_otherwise = 2 * (beta_safe / alpha_safe) * (
        torch.pow(x / beta_safe + 1., 0.5 * alpha) - 1.)
    loss_otherwise_d1 = torch.pow(x / beta_safe + 1., 0.5 * alpha - 1.)

    # Select which of the cases of the loss to return.
    loss = torch.where(
        alpha == 0, loss_zero,
        torch.where(alpha == 2, loss_two, loss_otherwise))
    loss_d1 = torch.where(
        alpha == 0, loss_zero_d1,
        torch.where(alpha == 2, loss_two_d1, loss_otherwise_d1))

    return loss, loss_d1, torch.zeros_like(x)


def homography_error(T, T_gt, H, W):
    """Compute the corner error between batched homographies.
    Use the 4-point parameterization.
    Args:
        T, T_gt: (batched) homographies of size (..., 3, 3).
        H, W: height and width of the image.
    """
    z = torch.zeros_like(H)
    corners0 = torch.stack([
        torch.stack([z, z], -1),
        torch.stack([W-1, z], -1),
        torch.stack([W-1, H-1], -1),
        torch.stack([z, H-1], -1)], -2).float()
    corners1_gt = from_homogeneous(
        to_homogeneous(corners0) @ T_gt.transpose(-1, -2))
    corners1 = from_homogeneous(to_homogeneous(corners0) @ T.transpose(-1, -2))
    d = ((corners1 - corners1_gt)**2).sum(-1)
    return d.mean(-1)


def pose_error(R1, t1, R2, t2):
    """Compute the rotation and translation errors of a batch of poses.
    Args:
        R1, t1: rotation matri and translation vector of camera 1.
        R2, t2: same for camera 2.
    Returns:
        A tuple of rotation (in deg) and translation errors.
    """
    dt = torch.norm(t1 - t2, p=2, dim=-1)
    trace = torch.diagonal(R1.transpose(-1, -2) @ R2, dim1=-1, dim2=-2).sum(-1)
    cos = torch.clamp((trace - 1) / 2, -1, 1)
    dr = torch.acos(cos).abs() / math.pi * 180
    return dr, dt
