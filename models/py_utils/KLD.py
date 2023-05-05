# import torch
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
import torch

def xy_wh_r_2_xy_sigma(xywhr):

    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)
    sigma = R.bmm(S.square()).bmm(R.permute(0, 2, 1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma

def try_vis():
    theta = [math.pi / 6]
    mean = [1, 1]
    wh = [3, 2]

    xy, sigma = xy_wh_r_2_xy_sigma(torch.Tensor([mean+wh+theta]))

    xy = list(xy.reshape(xy.shape[1:]).numpy())
    sigma = list(sigma.reshape(sigma.shape[1:]).numpy())

    x, y = np.mgrid[-4:4:.01, -4:4:.01]
    pos = np.dstack((x, y))
    rv = multivariate_normal(xy, sigma)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.contourf(x, y, rv.pdf(pos), 50)
    plt.show()




def kld_loss(pred, target, alpha=1.0, sqrt=True):
    """Kullback-Leibler Divergence loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        alpha (float): Defaults to 1.0.
        sqrt (bool): Whether to sqrt the distance. Defaults to True.

    Returns:
        loss (torch.Tensor)
    """
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    _shape = xy_p.shape

    xy_p = xy_p.reshape(-1, 2)

    xy_t = xy_t.reshape(-1, 2)
    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)

    Sigma_p_inv = torch.stack((Sigma_p[..., 1, 1], -Sigma_p[..., 0, 1],
                               -Sigma_p[..., 1, 0], Sigma_p[..., 0, 0]),
                              dim=-1).reshape(-1, 2, 2)
    Sigma_p_inv = Sigma_p_inv / (1e-25+Sigma_p.det()).unsqueeze(-1).unsqueeze(-1)

    dxy = (xy_p - xy_t).unsqueeze(-1)
    xy_distance = 0.5 * dxy.permute(0, 2, 1).bmm(Sigma_p_inv).bmm(dxy).view(-1)

    whr_distance = 0.5 * Sigma_p_inv.bmm(Sigma_t).diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    Sigma_p_det_log = Sigma_p.det().log()
    Sigma_t_det_log = Sigma_t.det().log()
    whr_distance = whr_distance + 0.5 * (Sigma_p_det_log - Sigma_t_det_log)
    whr_distance = whr_distance - 1
    distance = (xy_distance / (alpha * alpha) + whr_distance)
    if sqrt:
        distance = torch.sqrt(distance.clamp(0)+1e-30)

    distance = distance.reshape(_shape[:-1])
 
    return distance

def jd_loss(pred, target, alpha=1.0, sqrt=True):
    """Symmetrical Kullback-Leibler Divergence loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
        alpha (float): Defaults to 1.0.
        sqrt (bool): Whether to sqrt the distance. Defaults to True.

    Returns:
        loss (torch.Tensor)
    """
    jd = kld_loss(
        pred,
        target,
        alpha=alpha,
        sqrt=False,
        )
    jd = jd + kld_loss(
        target,
        pred,
        alpha=alpha,
        sqrt=False,
        )
    jd = jd * 0.5
    if sqrt:
        jd = torch.sqrt(jd.clamp(0)+1e-30)
    jd = torch.log1p(jd+1e-30)
    return 1 - 1 / (1+ jd)

