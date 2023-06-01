def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    # 配置模型设置,读取txt文件
    parser.add_argument('-c','--config', is_config_file=True,
                        help='config file path')

    ### 模型配置设置
    # 本次实验的名称,作为log中文件夹的名字
    parser.add_argument("--proname", type=str,default="nerf_demo_lego",
                        help='project name')

    # 输出目录, 保存模型以及日志
    parser.add_argument("--outdir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    # 指定数据集的目录,根据论文文件类别,数据集分别为合成数据集，以及真是数据集
    parser.add_argument("--datadir", type=str, default='./data/nerf_synthetic/lego',
                        help='input data directory')

    ### 训练配置
    # 中心裁剪的训练轮数
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # 数据格式
    parser.add_argument("--dataset_type", type=str, default='blender',
                        help='options: llff / blender')

    # 对于较大的数据集，分割test和val数据集，只使用其中的一部分数据
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')


    ## blender flags
    # 白色背景 由于合成数据集聚焦的重点是目标实例,没有边界效应
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    # 使用一半分辨率 用于合成数据集的训练
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ### llff flags
    # downsampling 下采样的倍率因子
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')

    # ndc 空间转换
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')

    # 使用线性视差代替深度
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')

    # 设置球形360场景
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')

    # 数据分割同上，采用论文倍率
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    # log输出的频率
    parser.add_argument("--log_print_f", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--log_img_f", type=int, default=500,
                        help='frequency of tensorboard image logging')
    # 保存模型的频率
    # 每隔1w次迭代保存一个
    parser.add_argument("--model_weights_f", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    # 执行测试集渲染的频率
    parser.add_argument("--model_testset_f", type=int, default=50000,
                        help='frequency of testset saving')
    # 执行渲染视频的频率
    parser.add_argument("--video_f", type=int, default=50000,
                        help='frequency of render_poses video saving')

    ### 训练配置
    # 这里的batch size，指的是光线的数量,像素点的数量 参考 1024
    parser.add_argument("--batchsz", type=int, default=32 * 32,
                        help='batch size (number of random rays per gradient step)')
    # 学习率 设置为论文学习率
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    # 学习率衰减
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    # 网络并行训练的ray数量
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')

    # 网络并行处理点的数量
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')

    # coarse网络 全连接层数
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    # 网络维度
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')

    # fine网络  全连接层数
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')

    # 网络维度
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')


    # 合成的数据集一般都是True, 每次只从一张图片中选取随机光线
    # 真实的数据集一般都是False, 图形先混在一起
    parser.add_argument("--use_batch", action='store_true',default=False,
                        help='only take random rays from 1 image at a time')

    # 不加载权重,重新开始训练
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')

    # 粗网络的权重文件的位置
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    ### 分层采样(hierarchy sampling)设置 根据论文设置
    # coarse网络采样点数
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    # fine网络采样点数
    parser.add_argument("--N_importance", type=int, default=128,
                        help='number of additional fine samples per ray')

    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')

    # 不适用视角数据 进行消融实验
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')

    ### 位置编码(position encoding) 根据论文设置
    # 0 使用位置编码，-1 不使用位置编码
    parser.add_argument("--is_embed", type=bool, default=True,
                        help='set 0 for default positional encoding, -1 for none')

    # L=10 coarse network 3*2*10+3 = 63
    parser.add_argument("--multires_position", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')

    # L=4 fine network 3*2*4+3 = 27
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')

    # 添加高斯噪声提高模型性能
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    # 仅进行渲染
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    # 渲染test数据集
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    # 下采样的倍数 由于真实图片尺寸太大 对其进行下采样
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    return parser

parser = config_parser()
args = parser.parse_args()
print(args)
