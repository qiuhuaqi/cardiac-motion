"""Loss functions"""
import torch


def diffusion_loss(dvf):
    """
    Calculate diffusion loss as a regularisation on the displacement vector field (DVF)

    Args:
        dvf: (Tensor of shape (N, 2, H, W)) displacement vector field estimated

    Returns:
        diffusion_loss_2d: (Scalar) diffusion regularisation loss
        """

    # spatial derivatives
    dvf_dx = dvf[:, :, 1:, 1:] - dvf[:, :, :-1, 1:]  # (N, 2, H-1, W-1)
    dvf_dy = dvf[:, :, 1:, 1:] - dvf[:, :, 1:, :-1]  # (N, 2, H-1, W-1)
    return (dvf_dx.pow(2) + dvf_dy.pow(2)).mean()


def huber_loss_spatial(dvf):
    """
    Calculate approximated spatial Huber loss
    Args:
        dvf: (Tensor of shape (N, 2, H, W)) displacement vector field estimated

    Returns:
        loss: (Scalar) Huber loss spatial

    """
    eps = 1e-8 # numerical stability

    # spatial derivatives
    dvf_dx = dvf[:, :, 1:, 1:] - dvf[:, :, :-1, 1:]  # (N, 2, H-1, W-1)
    dvf_dy = dvf[:, :, 1:, 1:] - dvf[:, :, 1:, :-1]  # (N, 2, H-1, W-1)
    return ((dvf_dx.pow(2) + dvf_dy.pow(2)).sum(dim=1) + eps).sqrt().mean()


def huber_loss_temporal(dvf):
    """
    Calculate approximated temporal Huber loss

    Args:
        dvf: (Tensor of shape (N, 2, H, W)) displacement vector field estimated

    Returns:
        loss: (Scalar) huber loss temporal

    """
    eps = 1e-8  # numerical stability

    # magnitude of the dvf
    dvf_norm = torch.norm(dvf, dim=1)  # (N, H, W)

    # temporal derivatives, 1st order
    dvf_norm_dt = dvf_norm[1:, :, :] - dvf_norm[:-1, :, :]
    loss = (dvf_norm_dt.pow(2) + eps).sum().sqrt()
    return loss
