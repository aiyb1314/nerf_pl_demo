# -*- coding:utf-8 -*-
# @Time    : 2023/2/10 19:56
# @Author  : xj

import torch
import numpy as np

# 误差的计算
img2mse = lambda x, y: torch.mean((x - y) ** 2)

mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

