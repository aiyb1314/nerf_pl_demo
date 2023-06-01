import os
import torch
import numpy as np
import imageio
import json
import cv2

### 相机坐标->图像坐标->世界坐标 转换
# 平移
trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

print(f'trans_t: {trans_t}')

# 绕x轴的旋转
rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

# 绕y轴的旋转
rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()

print(f'trans_t: {trans_t},{rot_phi},{rot_theta}')


# 相机坐标->世界坐标
def pose_spherical(theta, phi, radius):
    """
    theta: -180 -- +180，间隔为9
    phi: 固定值 -30
    radius: 固定值 4
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_blender_data(datadir, half_res=False, testskip=1):
    """
    testskip: test和val数据集，只会读取其中的一部分数据，跳着读取
    """
    data_splits = ['train', 'val', 'test']
    # 存储了三个json文件的数据
    # 读取三种数据集 meta存储相机参数 camera_angle_x frame:{file_path,rotation,transform_matrix
    data_lst = {}
    for s in data_splits:
        with open(os.path.join(datadir, 'transforms_{}.json'.format(s)), 'r') as fp:
            data_lst[s] = json.load(fp)
    all_imgs = []
    all_poses = []
    counts = [0]
    for dn in data_splits:
        data = data_lst[s]
        imgs = []
        poses = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            # 测试集如果数量很多，可能会设置testskip
            skip = testskip
        # 读取所有的图片，以及所有对应的transform_matrix skip表示间隔取值
        for frame in data['frames'][::skip]:
            fpath = os.path.join(datadir, frame['file_path'] + '.png')
            img = imageio.v2.imread(fpath)
            imgs.append(img)
            poses.append(np.array(frame['transform_matrix']))
        # 归一化 数据处理
        imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)，4通道 rgba
        poses = np.array(poses).astype(np.float32)
        # 用于计算train val test的递增值
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    # train val test 三个list
    data_split_idx = [np.arange(counts[i], counts[i + 1]) for i in range(len(data_splits))]
    # train test val 拼一起
    imgs = np.concatenate(all_imgs, axis=0)
    poses = np.concatenate(all_poses, axis=0)

    H, W = imgs[0].shape[:2] # 获取图像长宽
    # train test val camera_angle_x 这个变量值是相同的
    camera_angle_x = float(data['camera_angle_x'])
    # 计算焦距
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    #  np.linspace(-180, 180, 40 + 1) 9度一个间隔
    # (40,4,4), 渲染的结果就是40帧 渲染结果
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    # 合成数据集 分辨率减半
    if half_res:
        H = H // 2
        W = W // 2
        # 焦距一半
        focal = focal / 2.
        # 初始化 RGBA矩阵
        # imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        # resize interpolation 内插函数
        # for i, img in enumerate(imgs):
        #     # 调整成一半的大小
        #     imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        # imgs = imgs_half_res cv2: [w,h] => numpy h w
        half_imgs = [cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA) for img in imgs]
        return half_imgs, poses, render_poses, [H, W, focal], data_split_idx
    else:
        return imgs, poses, render_poses, [H, W, focal], data_split_idx

