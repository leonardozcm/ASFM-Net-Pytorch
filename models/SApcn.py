import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pcn import FeatureExtractor as Encoder
from models.pcn import Decoder
from snowmodels.model import SPD
from models.modelutils import *


class ASFMRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.coarse_mlp = MLP([1024, 1024, 1024, 512 * 3])
        self.mean_fc = nn.Linear(1024, 128)
        self.up_branch_mlp_conv_mf = MLPConv([1157, 128, 64])
        self.up_branch_mlp_conv_nomf = MLPConv([1029, 128, 64])
        self.contract_expand = ContractExpandOperation(64, 2)
        self.fine_mlp_conv = MLPConv([64, 512, 512, 3])

    def forward(self, code, level0, step_ratio=4):
        '''
        :param code: B * C
        :param inputs: B * C * N
        :param step_ratio: int
        :param num_extract: int
        :param mean_feature: B * C
        :return: coarse(B * N * C), fine(B, N, C)
        '''
        # coarse = torch.tanh(self.coarse_mlp(code))  # (32, 1536)
        # coarse = coarse.view(-1, 512, 3)  # (32, 512, 3)
        # coarse = coarse.transpose(2, 1).contiguous()  # (32, 3, 512)

        # inputs_new = inputs.transpose(2, 1).contiguous()  # (32, 2048, 3)
        # input_fps = symmetric_sample(
        #     inputs_new, int(num_extract/2))  # [32, 512,  3]
        # input_fps = input_fps.transpose(2, 1).contiguous()  # [32, 3, 512]
        # level0 = torch.cat([input_fps, coarse], 2)   # (32, 3, 1024)

        for i in range(int(math.log2(step_ratio))):
            num_fine = 2 ** (i + 1) * 1024
            grid = gen_grid_up(2 ** (i + 1)).cuda().contiguous()
            grid = torch.unsqueeze(grid, 0)   # (1, 2, 2)
            grid_feat = grid.repeat(level0.shape[0], 1, 1024)   # (32, 2, 2048)
            point_feat = torch.unsqueeze(level0, 3).repeat(
                1, 1, 1, 2)  # (32, 3, 1024, 2)
            point_feat = point_feat.view(-1, 3, num_fine)  # (32, 3, 2048)
            global_feat = torch.unsqueeze(code, 2).repeat(
                1, 1, num_fine)  # (32, 1024, 2048)

            feat = torch.cat([grid_feat, point_feat, global_feat], dim=1)
            # (32, 64, 2048)
            feat1 = F.relu(self.up_branch_mlp_conv_nomf(feat))

            feat2 = self.contract_expand(feat1)  # (32, 64, 2048)
            feat = feat1 + feat2  # (32, 64, 2048)

            fine = self.fine_mlp_conv(feat) + point_feat  # (32, 3, 2048)
            level0 = fine

        return fine.transpose(1, 2).contiguous()


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
        grid = gen_grid_up(2 ** (self.i + 1)).permute(1, 0).to(level0.device)
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
    def __init__(self, dim_feat=512, num_pc=256, num_p0=512, radius=1, up_factors=None):
        super(ASFM, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = [1] + up_factors

        uppers = []
        for i, factor in enumerate(up_factors):
            uppers.append(
                SPD(dim_feat=dim_feat, up_factor=factor, i=i, radius=radius))

        self.uppers = nn.ModuleList(uppers)

    def forward(self, x):
        '''
         x(bs,3.N)

         returns:
         v(bs, 1024)
         y_coarse(bs, 4096, 3)
         y_fine(bs, 4096, 3)
        '''
        feat = self.encoder(x)
        _, y_coarse = self.decoder(feat)

        y_coarse = y_coarse.permute(0, 2, 1).contiguous()
        arr_pcd = [y_coarse]
        # print("y_coarse: ", y_coarse.shape)  # (bs, 4096, 3)

        # X-Y plane mirro sample
        coarse_fps = fps_subsample(y_coarse, 512)  # (bs, 512, 3)
        # print("coarse_fps: ", coarse_fps.shape)
        inputs_fps = symmetric_sample(
            x.permute(0, 2, 1).contiguous(), 512 // 2)  # (bs, 512, 3)
        # print("inputs_fps: ", y_coarse.shape)

        pcd = torch.cat([coarse_fps, inputs_fps], dim=1)  # (bs, 1024, 3)

        K_prev = None
        pcd = pcd.permute(0, 2, 1).contiguous()
        for upper in self.uppers:
            pcd, K_prev = upper(pcd, feat, K_prev)
            arr_pcd.append(pcd.permute(0, 2, 1).contiguous())

        # print("y_fine: ", y_fine.shape)
        return feat, arr_pcd

# class ASFM(nn.Module):
#     def __init__(self, step_ratio=4):
#         super(ASFM, self).__init__()

#         self.encoder = Encoder()
#         self.decoder = Decoder()
#         self.refiner = ASFMRefiner()
#         self.step_ratio = step_ratio

#     def forward(self, x):
#         '''
#          x(bs,3.N)

#          returns:
#          v(bs, 1024)
#          y_coarse(bs, 4096, 3)
#          y_fine(bs, 4096, 3)
#         '''
#         v = self.encoder(x)
#         _, y_coarse = self.decoder(v)
#         y_coarse = y_coarse.permute(0, 2, 1).contiguous()
#         # print("y_coarse: ", y_coarse.shape)  # (bs, 4096, 3)

#         # X-Y plane mirro sample
#         coarse_fps = fps_subsample(y_coarse, 512)  # (bs, 512, 3)
#         # print("coarse_fps: ", coarse_fps.shape)
#         inputs_fps = symmetric_sample(
#             x.permute(0, 2, 1).contiguous(), 512 // 2)  # (bs, 512, 3)
#         # print("inputs_fps: ", y_coarse.shape)

#         y_fine = torch.cat([coarse_fps, inputs_fps], dim=1)  # (bs, 1024, 3)

#         y_fine = self.refiner(v, y_fine.permute(
#             0, 2, 1).contiguous(), self.step_ratio)  # (bs, 4096, 3)

#         # print("y_fine: ", y_fine.shape)
#         return v, y_coarse, y_fine


if __name__ == "__main__":
    pcs = torch.rand(16, 3, 2048).cuda()
    ae = ASFM(up_factors=[2, 2]).cuda()
    # ae_decoder = ASFMDecoder().cuda()

    v, arr_pcd = ae(pcs)

    print("v shape", v.size())
    print("1024:", arr_pcd[0].size())
    print("2048:", arr_pcd[1].size())
    print("4096:", arr_pcd[2].shape[1])
