import torch
import numpy as np
import torch.nn.functional as F

from utils.rGenerator import  get_rays, ndc_rays
from utils.hSampling import fine_sampling,coarse_sampling
DEBUG = False

def render(H, W, K,
           chunk=1024 * 32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    if c2w is not None:
        # 坐标转换
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        # 光线的起始位置, 方向
        rays_o, rays_d = rays # [batchsz,3]

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        # 静态相机 相机坐标到世界坐标的转换
        if c2w_staticcam is not None:
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        # 单位向量 [batchsz,3]
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    # Create ray batch
    rays_o = rays_o.reshape([-1, 3]).float()
    rays_d = rays_d.reshape([-1, 3]).float()

    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # [bs,1],[bs,1]
    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])

    # 8=3+3+1+1 [batchsz,8]
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    if use_viewdirs:
        # 加了direction的三个坐标
        # [batchsz,11]
        rays = torch.cat([rays, viewdirs], -1)  # [bs,11]

    # Render and reshape
    # rgb_map,acc_map,raw,rbg_coarse,acc_coarse,z_std
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        # 对所有的返回值进行reshape
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    # 讲精细网络的输出单独拿了出来
    k_extract = ['rgb_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    # 前三是list，后5还是在map中
    return ret_list + [ret_dict]


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    # 将分批处理的结果拼接在一起
    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


# 这里面会经过神经网络
def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    pts_coarse,viewdirs, z_vals_coarse, rays_d, rays_o =  coarse_sampling(ray_batch,N_samples,lindisp,lindisp,perturb)
    # 使用神经网络 viewdirs [batchsz,3], network_fn 指的是粗糙NeRF或者精细NeRF
    # raw [bs,64,3]
    train_val_coarse = network_query_fn(pts_coarse, viewdirs, network_fn)

    # rgb值，权重的和，weights就是论文中的那个Ti*alpha
    rgb_map, acc_map, weights = raw2outputs(train_val_coarse, z_vals_coarse, rays_d, raw_noise_std, white_bkgd,
                                                             pytest=pytest)
    rgb_map_coarse, acc_map_coarse = rgb_map, acc_map
    ret = {}
    train_val_fine = None
    # 精细网络部分
    if N_importance > 0:
        run_fn = network_fn if network_fine is None else network_fine
        pts_fine,z_vals_fine = fine_sampling(z_vals_coarse,rays_o,rays_d,weights,N_importance,perturb,pytest)
        # viewdirs 与粗糙网络是相同的
        train_val_fine = network_query_fn(pts_fine, viewdirs, run_fn)
        rgb_map, acc_map, weights = raw2outputs(train_val_fine, z_vals_fine, rays_d, raw_noise_std, white_bkgd,
                                                                     pytest=pytest)
        ret['rgb_coarse'] = rgb_map_coarse
        ret['acc_coarse'] = acc_map_coarse
    ret['rgb_map'] = rgb_map
    ret['acc_map'] = acc_map
    if retraw:
        # 如果是两个网络，那么这个raw就是最后精细网络的输出
        ret['model_val'] = train_val_fine if train_val_fine is not None else train_val_coarse
    return ret


def raw2alpha(raw, dists, act_fn = F.relu):
    return 1. - torch.exp(-act_fn(raw) * dists)


# 模型训练值的转换
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    # Alpha的计算
    # relu, 负数拉平为0
    # raw2alpha = lambda raw, dists, act_fn=F.relu : 1. - torch.exp(-act_fn(raw) * dists)
    # [bs,63]
    # 采样点之间的距离
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # rays_d[...,None,:] [bs,3] -> [bs,1,3]
    # 1维 -> 3维 [4096,64]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    # RGB经过sigmoid处理
    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std
        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)
    # 计算公式3 [bs, 64],
    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

    # 后面这部分就是Ti，前面是alpha，这个就是论文上的那个权重w [bs,64] Ti = (1-alpha1)(1-alpha2)...(1-alphai)alphai
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)),
                                               1. - alpha + 1e-10], -1),-1)[:, :-1]
    # [bs, 64,1] * [bs,64,3]
    # 在第二个维度，64将所有的点的值相加 -> [32,3]
    # 公式3的结果值
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    # 权重和
    # 这个值仅做了输出用，后续并无使用
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, acc_map, weights
