# -*- coding:utf-8 -*-
# @Time    : 2023/1/30 15:54
# @Author  : xj

from opts import config_parser
import imageio
import time
from tqdm import tqdm
from utils.saveFuncs import *
from datasets.load_llff import load_llff_data
from datasets.load_blender import load_blender_data
from utils.vRender import *
from utils.inference import render_path
from utils.createFuncs import *
from utils.lossF import *
from utils.renderOnly import run_render_test
from utils.rGenerator import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num = torch.random.seed()
DEBUG = False

def train():
    # 解析参数
    parser = config_parser()
    args = parser.parse_args()
    print(args)
    # --------------------------------------------------------------------------------------------------------

    # Load data
    K = None


    # llff Local Light Field Fusion 局部光场融合
    if args.dataset_type == 'llff':
        images, transform_poses, bds, render_poses, data_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = transform_poses[0, :3, -1]
        transform_poses = transform_poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(data_test, list):
            data_test = [data_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            data_test = np.arange(images.shape[0])[::args.llffhold]

        idx_val = data_test
        data_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in data_test and i not in idx_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        # images，所有的图片，train val test在一起，poses也一样
        images, transform_poses, render_poses, hwf, data_split_idx = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape,hwf)
        # 数据分割 [0,25] [25,50] [50,75]
        data_train, data_val, data_test = data_split_idx

        # ray [near,far] 定义
        near = 2
        far = 6

        # 透明色处理 将 RGBA 转换成 RGB 图像
        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # 内参转换
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    center_pos = (H/2,W/2)
    # K 相机内参 focal 是焦距，(0.5h,0.5w) 是中心点坐标
    # 这个矩阵是相机坐标到图像坐标转换使用 如果没有指定，由计算得到
    if K is None:
        K = np.array([
            [focal, 0, center_pos[0]],
            [0, focal, center_pos[1]],
            [0, 0, 1]
        ])
    # --------------------------------------------------------------------------------------------------------

    # 使用测试集的transform_pose，而不是用那个固定生成的render_poses
    if args.render_test:
        render_poses = np.array(transform_poses[data_test])

    # 使用GPU加速训练
    render_poses = torch.Tensor(render_poses).to(device)

    # --------------------------------------------------------------------------------------------------------

    # 创建log文件夹

    outdir = args.outdir
    proname = args.proname
    print(f'模型保存路径: {outdir},{proname}')
    create_log_files(outdir, proname, args)

    # --------------------------------------------------------------------------------------------------------

    # 创建模型
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf_model(args)

    # 有可能从中间迭代恢复运行的
    global_step = start

    break_pro = {
        'near': near,
        'far': far,
    }

    render_kwargs_train.update(break_pro)
    render_kwargs_test.update(break_pro)

    # --------------------------------------------------------------------------------------------------------

    # 这里会使用render_poses
    if args.render_only:
        # 仅进行渲染，不进行训练
        print('RENDER ONLY')
        run_render_test(args, images, data_test, outdir, proname, render_poses, hwf, K, render_kwargs_test, start)
        return

    # --------------------------------------------------------------------------------------------------------

    # 如果对随机射线进行批处理，则准备射线批处理张量
    batchsz = args.batchsz

    use_batch =  args.use_batch

    if use_batch:
        # get_rays->(rays_o, rays_d)
        rays = np.stack([get_rays(H, W, K, p) for p in transform_poses], 0)  # [N, ro+rd, H, W, chanel] [75,2,800,800,3]
        rays_rgb = np.concatenate([rays, images[:, None]], 1).transpose([0, 2, 3, 1, 4]) # [N, ro+rd+rgb, H, W, 3] [75,3,800,800,3] -> [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in data_train], 0)  # train images only, 仅使用训练文件夹下的数据
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3]).astype(np.float32)  # [N*H*W, ro+rd+rgb, 3] => [25*800*800,]
        np.random.shuffle(rays_rgb)  # 打乱光线
        print(f"shuffle ray according to the paper")
        images = torch.Tensor(images).to(device)
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    transform_poses = torch.Tensor(transform_poses).to(device)
    # --------------------------------------------------------------------------------------------------------
    # Train
    N_iters = 20000 + 1
    start = start + 1
    idx = 0
    for i in tqdm(range(start, N_iters)):
        time0 = time.time()
        # Sample random ray batch
        if use_batch:
            # Random over all images
            # 一批光线
            batch = rays_rgb[idx:idx + batchsz]  # [batchsz,3,3]

            batch = torch.transpose(batch, 0, 1) # [3,batchsz,3]
            batch_rays, target_s = batch[:2], batch[2]  # 前两个是rays_o和rays_d, [2,batchsz,3]  第三个是target就是image的rgb [rgb,batchsz,3]

            idx += batchsz

            if idx >= rays_rgb.shape[0]:
                # 所有光线用过之后，重新打乱
                # rays_rgb = torch.random.shuffle(rays_rgb) pytorch 中的 shuffle 函数会出现重复问题
                # 采用打扰数组索引的方式进行重新排列
                rand_idx = np.random.shuffle(np.arange(rays_rgb.shape[0]))
                rays_rgb = rays_rgb[rand_idx]
                idx = 0

        else:
            # 随机选择一张图片
            img_idx = np.random.choice(data_train)
            target = images[img_idx]  # [400,400,3] 图像内容

            target = torch.Tensor(target).to(device)
            # 获取转换矩阵
            transform_pose = transform_poses[img_idx]

            if batchsz is not None:
                rays_o, rays_d = get_rays(H, W, K, transform_pose)  # [1,800,800,3]
                # 中心裁剪迭代次数
                if i < args.precrop_iters:
                    dH = int(H // 2 * args.precrop_frac)
                    dW = int(W // 2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                            torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW), indexing='ij',
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H),
                                                        torch.linspace(0, W - 1, W), indexing='ij'),
                                         -1)  # (H, W, 2)

                coords = coords.reshape([-1, 2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[batchsz], replace=False)  # (batchsz,) 选择像素点 从 H*W 中选出 batchsz个像素点
                # 选出的像素坐标
                select_coords = coords[select_inds].long()  # (batchsz, 2)
                rays_o = torch.Tensor(rays_o)
                rays_d = torch.Tensor(rays_d)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]] # (batchsz, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]] # (batchsz, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)  # 堆叠 o和d [2,batchsz,3]
                # target 也同样选出对应位置的点
                # target 用来最后的mse loss 计算
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (batchsz, 3)

        #####  Core optimization loop  #####
        # rgb 网络计算出的图像
        # 前三是精细网络的输出内容，其他的还保存在一个dict中，有5项
        rgb, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                        verbose=i < 10, retraw=True,
                                        **render_kwargs_train)

        optimizer.zero_grad()
        # 计算loss
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        # 计算指标
        psnr = mse2psnr(img_loss)

        # rgb0 粗网络的输出
        if 'rgb_coarse' in extras:
            img_loss0 = img2mse(extras['rgb_coarse'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # 学习率衰减
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        # 保存模型
        if i % args.model_weights_f == 0:
            dic = {
                # 运行的轮次数目
                'global_step': global_step,
                # 粗网络的权重
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                # 精细网络的权重
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                # 优化器的状态
                'optimizer_state_dict': optimizer.state_dict(),
            }
            saveModelState(outdir,proname,dic,i)

        # 生成测试视频
        if i % args.video_f == 0 and i > 0:
            end = time.time()
            with torch.no_grad():
                rgbs = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            moviebase = os.path.join(outdir, proname, f'{end}_')
            # 360度转一圈的视频
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)

        # 执行测试，使用测试数据
        if i % args.model_testset_f == 0 and i > 0:
            testsavedir = os.path.join(outdir, proname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', transform_poses[data_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(transform_poses[data_test]).to(device), hwf, K, args.chunk, render_kwargs_test,
                            gt_imgs=images[data_test], savedir=testsavedir)
            print('Saved test set')

        # 用时
        dt = time.time() - time0
        # 打印log信息的频率
        if i % args.log_print_f == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()} Time: {dt}")

        global_step += 1