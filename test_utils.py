import tensorflow as tf
from utils.io import IO
import open3d
from models.utils import symmetric_sample, gen_grid_up
import os
import numpy as np
import torch
import math


def write_tensor2pcd(cloud_tensor):
    cloud_tensor = cloud_tensor.squeeze(dim=0)
    cloud_tensor = cloud_tensor.to(
        torch.device('cpu')).squeeze().detach().numpy()

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(cloud_tensor)
    open3d.io.write_point_cloud(
        "visualization/output.pcd", pcd, write_ascii=True)


def standardize_output(ptcloud):

    non_zeros = torch.sum(ptcloud, dim=2).ne(0)
    # print(non_zeros.shape)  #torch.Size([1, 1572864])
    p = ptcloud[non_zeros].squeeze(dim=0)
    # print(p.shape)   #torch.Size([45816, 3])
    return p


def tf_gen_grid_up(up_ratio):
    sqrted = int(math.sqrt(up_ratio))+1
    for i in range(1, sqrted+1).__reversed__():
        if (up_ratio % i) == 0:
            num_x = i
            num_y = up_ratio//i
            break
    grid_x = tf.linspace(-0.2, 0.2, num_x)
    grid_y = tf.linspace(-0.2, 0.2, num_y)

    x, y = tf.meshgrid(grid_x, grid_y)
    grid = tf.reshape(tf.stack([x, y], axis=-1),
                      [-1, 2])  # [2, 2, 2] -> [4, 2]
    return grid

# test symmetric

# file_path = '/home/chriskafka/dataset/ShapeNetCompletion/train/partial/02691156/1a74b169a76e651ebc0909d98a1ff2b4/02.pcd'
# data = IO.get(file_path=file_path).astype(np.float32)
# data = np.expand_dims(data, axis=0)
# data_tensor = torch.from_numpy(data).cuda()
# sym_pcd = symmetric_sample(data_tensor, 512)
# print(sym_pcd.shape)
# write_tensor2pcd(standardize_output(sym_pcd))

# test 2D grid


a = [1, 2, 3]
b = [4, 5, 6]

x = torch.tensor(a)
y = torch.tensor(b)

num = 8

print(gen_grid_up(num))
print(tf_gen_grid_up(num))
