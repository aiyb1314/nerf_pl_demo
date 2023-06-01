# -*- coding:utf-8 -*-
# @Time    : 2023/2/1 16:04
# @Author  : xj
import os
from utils.lossF import *
from utils.inference import render_path
import imageio

def run_render_test(args, images, i_test, outdir, proname, render_poses, hwf, K, render_kwargs_test, start):
    with torch.no_grad():
        if args.render_test:
            # render_test switches to test poses
            images = images[i_test]
        else:
            # Default is smoother render_poses path
            images = None

        testsavedir = os.path.join(outdir, proname,
                                   'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', render_poses.shape)

        rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images,
                              savedir=testsavedir, render_factor=args.render_factor)
        print('Done rendering', testsavedir)
        imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)