# -*- coding:utf-8 -*-
# @Time    : 2023/2/5 9:34
# @Author  : xj

import os
import torch
def saveModelState(outdir,proname,dict,i):
    path = os.path.join(outdir, proname, '{:06d}.tar'.format(i))
    torch.save(dict, path)
    print('Saved checkpoints at', path)