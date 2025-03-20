import torch
from torchvision import transforms as tvf
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import cv2

from model.semantic_masker.DeepLabV3Plus.datasets import Cityscapes


def patch_selection(imgs_path, seg_results_path):
    device = torch.device("cuda")
    imgs_list = os.listdir(imgs_path)
    for im in imgs_list:
        img_path = os.path.join(imgs_path, im)
        # 读取原始图像
        img = Image.open(img_path).convert('RGB')
        # 读取语义分割结果
        seg_result_path = os.path.join(seg_results_path, "%s.npy" % im.split(".")[0])
        seg_result = np.load(seg_result_path)

        base_tf = tvf.Compose([
            tvf.ToTensor(),
        ])
        img_pt = base_tf(img).to(device)
        seg_result_pt = base_tf(seg_result).to(device)
        # 对图像进行裁剪，使得图像可被划分为14×14的patch
        c, h, w = img_pt.shape
        h_new, w_new = (h // 14) * 14, (w // 14) * 14
        h_num, w_num = h // 14, w // 14
        img_pt = tvf.CenterCrop((h_new, w_new))(img_pt)[None, ...]
        seg_result_pt = tvf.CenterCrop((h_new, w_new))(seg_result_pt)[None, ...]
        print(seg_result_pt.shape)
        # 定义patch的大小
        patch_size = (14, 14)
        # 使用unfold函数按照patch大小统计值
        unfolded_tensor = seg_result_pt.unfold(2, patch_size[0], patch_size[0]).unfold(3, patch_size[1], patch_size[1])
        print(unfolded_tensor.shape)
        # 统计每个patch内数量最多的值
        patch_mode, _ = unfolded_tensor.contiguous().view(seg_result_pt.size(0), seg_result_pt.size(1), -1,
                                                          patch_size[0] * patch_size[1]).mode(dim=3)
        # 可视化
        patch_mode = patch_mode.squeeze(dim=0).squeeze(dim=0)
        patch_mode = patch_mode.cpu()
        patch_mode = patch_mode.numpy()
        print(patch_mode.shape)
        # decode_fn = Cityscapes.decode_target
        # patch_mode_color = decode_fn(patch_mode).astype('uint8')
        # mask = cv2.resize(patch_mode_color, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
        # plt.imshow(np.transpose(np.array(img_pt.squeeze(dim=0).cpu()), (1, 2, 0)))
        # plt.imshow(mask, alpha=0.7)
        # plt.show()
        # 筛选静态目标的patch
        static_indices = np.where(patch_mode < 11)
        static_patch = np.zeros_like(patch_mode)
        static_patch[static_indices] = 1
        static_patch = static_patch.reshape(h_num, w_num)
        color_array = np.zeros((static_patch.shape[0], static_patch.shape[1], 3))
        color_array[static_patch == 1] = [0, 0, 1]  # 蓝色
        color_array[static_patch == 0] = [1, 0, 0]  # 红色
        mask = cv2.resize(color_array, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
        plt.imshow(np.transpose(np.array(img_pt.squeeze(dim=0).cpu()), (1, 2, 0)))
        plt.imshow(mask, alpha=0.5)
        # 设置x和y轴的刻度
        x_ticks = range(0, w_new, 14)  # 0到100，步长为14
        y_ticks = range(0, h_new, 14)
        # 设置刻度
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)

        # 绘制网格线
        plt.grid(color='white', linestyle='-', linewidth=1)
        plt.show()


if __name__ == "__main__":
    print("Start")
    patch_selection(
        imgs_path="../test_imgs",
        seg_results_path="../class_maps"
    )
