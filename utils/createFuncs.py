# -*- coding:utf-8 -*-
# @Time    : 2023/2/8 15:56
# @Author  : xj
import os
from model import NeRF
from utils.positionEmbed import *
import torch


# 分批化，如果batchsz > memory 分批训练
def batchify(fn, chunk):
    """
    Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        # 以chunk分批进入网络，防止显存爆掉，然后在拼接
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """
    被下面的create_nerf 封装到了lambda方法里面
    Prepares inputs and applies network 'fn'.
    inputs: pts，光线上的点 如 [1024,64,3]，1024条光线，一条光线上64个点
    viewdirs: 光线起点的方向
    fn: 神经网络模型 粗糙网络或者精细网络
    embed_fn:
    embeddirs_fn:
    netchunk:
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])  # [N_rand*64,3]
    # 坐标点进行编码嵌入 [N_rand*64,63]
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        # 方向进行位置编码
        embedded_dirs = embeddirs_fn(input_dirs_flat)  # [N_rand*64,27]
        embedded = torch.cat([embedded, embedded_dirs], -1)

    # 里面经过网络 [bs*64,3]
    outputs_flat = batchify(fn, netchunk)(embedded)
    # [bs*4,4] -> [bs,64,4]
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def create_log_files(outdir, proname, args):
    os.makedirs(os.path.join(outdir, proname), exist_ok=True)

    # 保存一份参数文件
    f = os.path.join(outdir, proname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    # 保存一份配置文件
    if args.config is not None:
        f = os.path.join(outdir, proname, 'config.ini')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    return outdir, proname


def create_nerf_model(args,device=None):
    """
        create NeRF's MLP model.
    """
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_fn, input_ch = get_embedder(args.multires_position, args.is_embed)

    input_ch_views = 0
    embeddirs_fn = None

    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.is_embed)

    # 想要=5生效，首先需要use_viewdirs=False and N_importance>0
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    # 粗网络
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)

    # 获取参数变量
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        # 精细网络
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        # 模型参数
        grad_vars += list(model_fine.parameters())

    # netchunk 是网络中处理的点的batch_size
    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                        embed_fn=embed_fn,
                                                                        embeddirs_fn=embeddirs_fn,
                                                                        netchunk=args.netchunk)

    # Create optimizer
    # 优化器
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    outdir = args.outdir
    proname = args.proname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(outdir, proname, f) for f in sorted(os.listdir(os.path.join(outdir, proname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)

    # load参数
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        # 精细网络
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        # 粗网络
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    print(model_fine)

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer
