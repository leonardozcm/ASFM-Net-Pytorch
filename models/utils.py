#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Xiang

from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
import torch
from torch import nn
import math


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), if_bn=True, activation_fn=torch.relu):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out


def symmetric_sample(points, num):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, num * 2, 3)
    """
    input_fps = fps_subsample(points, num)

    input_fps_flip = torch.cat(
        [torch.unsqueeze(input_fps[:, :, 0], dim=2), torch.unsqueeze(input_fps[:, :, 1], dim=2),
         torch.unsqueeze(-input_fps[:, :, 2], dim=2)], dim=2)
    input_fps = torch.cat([input_fps, input_fps_flip], dim=1)
    return input_fps


def gen_grid_up(up_ratio):
    '''
    returns
        tensor (up_ratio, 2)
    '''
    sqrted = int(math.sqrt(up_ratio))+1
    for i in range(1, sqrted+1).__reversed__():
        if (up_ratio % i) == 0:
            num_x = i
            num_y = up_ratio//i
            break
    grid_x = torch.linspace(-0.2, 0.2, num_x)
    grid_y = torch.linspace(-0.2, 0.2, num_y)

    x, y = torch.meshgrid(grid_x, grid_y, indexing='xy')
    grid = torch.reshape(torch.stack([x, y], dim=-1), [-1, 2])
    return grid


def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    new_pcd = gather_operation(pcd.permute(
        0, 2, 1).contiguous(), furthest_point_sample(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd
