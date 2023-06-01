# -*- coding:utf-8 -*-
# @Time    : 2023/2/10 19:57
# @Author  : xj

import torch
import numpy as np

# Hierarchical sampling

#common sampling
def coarse_sampling(
                ray_batch,
                N_samples,
                lindisp=False,
                perturb=0.,
                pytest=False):
    N_rays = ray_batch.shape[0]  # batchsz
    # 光线起始位置，光线的方向
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    # 视角的单位向量
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])  # [bs,1,2] near和far

    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # 采样点
    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)  # 插值采样
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
    # [batchsz,64] -> [batchsz,64]
    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples，64个采样点的中点
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)
        # [bs,64] 加上随机的噪声
        z_vals = lower + (upper - lower) * t_rand

    # 空间中的采样点
    # [batchsz, 64, 3]
    # 出发点+距离*方向 r(t) = o + td
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
    return pts,viewdirs,z_vals,rays_d,rays_o

def fine_sampling(z_vals,rays_o,rays_d,weights,N_importance=128,perturb=0.,pytest=False):
    # 粗糙网络的结果
    # 第二次计算mid，取中点位置 获取
    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples_fine = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
    z_samples_fine = z_samples_fine.detach()
    z_vals_fine, _ = torch.sort(torch.cat([z_vals, z_samples_fine], -1), -1)
    # 给精细网络使用的点
    # [N_rays, N_samples + N_importance, 3]
    pts_fine = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_fine[..., :, None]
    return pts_fine,z_vals_fine

def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    """
    bins: z_vals_mid
    """

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    # 归一化 [bs, 62]
    # 概率密度函数
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    # 累积分布函数
    cdf = torch.cumsum(pdf, -1)
    # 在第一个位置补0
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])  # [bs,128]

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF

    u = u.contiguous()
    # u 是随机生成的
    # 找到对应的插入的位置
    inds = torch.searchsorted(cdf, u, right=True)
    # 前一个位置，为了防止inds中的0的前一个是-1，这里就还是0
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    # 最大的位置就是cdf的上限位置，防止过头，跟上面的意义相同
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    # (batch, N_samples, 2)
    inds_g = torch.stack([below, above], -1)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # (batch, N_samples, 63)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    # 如[1024,128,63] 提取 根据 inds_g[i][j][0] inds_g[i][j][1]
    # cdf_g [1024,128,2]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    # 如上, bins 是从2到6的采样点，是64个点的中间值
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    # 差值
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    # 防止过小
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)

    t = (u - cdf_g[..., 0]) / denom

    # lower+线性插值
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


# ----------------------------------------------------------------------------------------------------------------------
