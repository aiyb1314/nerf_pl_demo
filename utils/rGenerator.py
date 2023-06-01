# -*- coding:utf-8 -*-
# @Time    : 2023/2/1 19:55
# @Author  : xj


import torch
import numpy as np

# Get Ray r = o + td
def get_rays(H, W, K, transform_matrix):
    """
    K：相机内参矩阵
    c2w: 相机到世界坐标系的转换
    """
    # 建立坐标系

    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H),indexing='xy')  # pytorch's meshgrid has indexing='ij'
    pts = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # rays_d = (800,800,3) transform_pose 源姿态 (source pose)
    transform_matrix = torch.Tensor(transform_matrix)


    # 光线原点坐标
    if torch.cuda.is_available():
        rays_d = torch.sum(pts[..., np.newaxis, :] * transform_matrix[:3, :3],
                           -1).cpu().numpy()  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        rays_o = transform_matrix[:3, -1].expand(rays_d.shape).cpu().numpy()
    else:
        rays_d = torch.sum(pts[..., np.newaxis, :] * transform_matrix[:3, :3],
                           -1).numpy()  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        rays_o = transform_matrix[:3, -1].expand(rays_d.shape).numpy()


    return rays_o, rays_d


#  ndc space rays generator
def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d