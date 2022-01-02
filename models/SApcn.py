import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pcn import Encoder
from models.utils import gen_grid_up, MLP_CONV, Conv2d


class ASFMDecoder(nn.Module):
    def __init__(self, num_coarse=1024, num_dense=16384):
        super(ASFMDecoder, self).__init__()

        self.num_coarse = num_coarse

        # fully connected layers
        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 3 * num_coarse)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)

    def forward(self, x):
        b = x.size()[0]
        # global features
        v = x  # (B, 1024)

        # fully connected layers to generate the coarse output
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        y_coarse = x.view(-1, 3, self.num_coarse)  # (B, 3, 1024)

        return y_coarse


class RefineUnit(nn.Module):
    def __init__(self, i=0) -> None:
        super(RefineUnit, self).__init__()

        self.i = i
        self.mlp_conv1 = MLP_CONV(in_channel=1029, layer_dims=(128, 64))
        self.relu1 = torch.nn.ReLU()

        self.conv2d_1 = Conv2d(64, 64, (1, 2))
        self.conv2d_2 = Conv2d(64, 128)
        self.conv2d_3 = Conv2d(64, 64)

        self.mlp_conv2 = MLP_CONV(in_channel=64, layer_dims=(512, 512, 3))

    def forward(self, level0, code):
        '''
        args:
        level0: (bs, N, 3)
        code: (bs, 1024)

        return:
        output:(bs, 2N, 3)
        '''
        num_fine = 2 ** (self.i + 1) * 1024
        # (2 ** (self.i + 1), 2)
        grid = gen_grid_up(2 ** (self.i + 1)).to(level0.device)
        grid = torch.unsqueeze(grid, 0)
        grid_feat = torch.tile(
            grid, (level0.shape[0], 1024, 1))  # (bs, num_fine, 2)
        point_feat = torch.tile(torch.unsqueeze(
            level0, 2), (1, 1, 2, 1))  # (bs, N, 2, 3)
        point_feat = torch.reshape(
            point_feat, (-1, num_fine, 3))  # (bs, 2*N, 3)
        global_feat = torch.tile(torch.unsqueeze(
            code, 1), (1, num_fine, 1))  # (bs, num_fine, 1024)

        feat = torch.cat([grid_feat, point_feat, global_feat],
                         dim=2)  # (bs, num_fine, 2+3+1024)

        feat = feat.permute(0, 2, 1)  # (bs, 2+3+1024, num_fine,)
        feat1 = self.mlp_conv1(feat)  # (bs, 64, num_fine)
        feat1 = self.relu1(feat1)

        feat2 = self.contract_expand_operation(feat1, 2)  # (bs, 64, num_fine)
        feat = feat1+feat2

        fine = self.mlp_conv2(feat2).permute(0, 2, 1)+point_feat

        return fine

    def contract_expand_operation(self, inputs, up_ratio):
        net = inputs
        net = torch.reshape(
            net, (net.shape[0], net.shape[1], -1, up_ratio))  # (bs, 64, num_fine / 2, 2)

        net = self.conv2d_1(net)   # (bs, 64, num_fine / 2, 1)

        net = self.conv2d_2(net)   # (bs, 128, num_fine / 2, 1)

        # (bs, 64, num_fine / 2, 2)
        net = torch.reshape(net,  (net.shape[0], 64, -1, up_ratio))

        net = self.conv2d_3(net)   # (bs, 64, num_fine/2 , 2)
        net = torch.reshape(net,  (net.shape[0], 64, -1))  # (bs, 64, num_fine)
        return net


class ASFM(nn.Module):
    def __init__(self, step_ratio=2):
        super(ASFM, self).__init__()

        self.encoder = Encoder()
        self.decoder = ASFMDecoder()
        self.refine_units = []
        for i in range(int(math.log2(step_ratio))):
            self.refine_units.append(RefineUnit(i))
        self.refine_units = nn.ModuleList(self.refine_units)

    def forward(self, x):
        v = self.encoder(x)
        y_coarse = self.decoder(v)
        y_fine = y_coarse
        for unit in self.refine_units:
            y_fine = unit(y_fine, v)

        return v, y_coarse.permute(0, 2, 1), y_fine


if __name__ == "__main__":
    pcs = torch.rand(16, 3, 2048)
    ae = ASFM(step_ratio=4)
    v, y_coarse, y_detail = ae(pcs)
    print(v.size(), y_coarse.size(), y_detail.size())
