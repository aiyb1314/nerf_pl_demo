# -*- coding:utf-8 -*-
# @Time    : 2023/2/1 19:52
# @Author  : xj

import torch
from torch import nn
# 位置编码的实现
class Embedder:
    def __init__(self, **kwargs):
        print(f'position encoding parameters:{kwargs} ')
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']  # 3
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            print(embed_fns)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        # 采样方式
        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                # sin(x),sin(2x),sin(4x),sin(8x),sin(16x),sin(32x),sin(64x),sin(128x),sin(256x),sin(512x)
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d
        self.embed_fns = embed_fns

        # 3D坐标是63，2D方向是27
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


# 位置编码相关
def get_embedder(multires, i=True):
    """
    multires: 3D 坐标是10，2D方向是4
    """
    if i == False:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    # 第一个返回值是lamda，给定x，返回其位置编码
    return embed, embedder_obj.out_dim
# ----------------------------------------------------------------------------------------------------------------------
