import torch
import torch.nn as nn
import torch.nn.functional as F
from snowmodels.utils import MLP_Res


class Encoder(nn.Module):
    def __init__(self, output_size=512):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, output_size, 1)

    def forward(self, x):
        batch_size, _, num_points = x.size()
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        global_feature, _ = torch.max(x, 2)
        x = torch.cat((x, global_feature.view(batch_size, -1,
                      1).repeat(1, 1, num_points).contiguous()), 1)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        global_feature, _ = torch.max(x, 2)
        return global_feature.view(batch_size, -1, 1)


class Decoder(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256):
        super(Decoder, self).__init__()
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128,
                             hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128,
                             hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        x1 = self.ps(feat)  # (b, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        # (b, 128, 256)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))
        completion = self.mlp_4(x3)  # (b, 3, 256)
        return completion


# class AutoEncoder(nn.Module):
#     def __init__(self):
#         super(AutoEncoder, self).__init__()

#         self.encoder = FeatureExtractor()
#         self.decoder = Decoder()

#     def forward(self, x):
#         v = self.encoder(x)
#         y_coarse, y_detail = self.decoder(v)
#         return v, y_coarse, y_detail


# if __name__ == "__main__":
#     pcs = torch.rand(16, 3, 4096).cuda()
#     encoder = Encoder().cuda()
#     v = encoder(pcs)
#     print(v.size())

#     decoder = Decoder().cuda()
#     y_c, y_d = decoder(v)
#     print(y_c.size(), y_d.size())

#     ae = AutoEncoder().cuda()
#     v, y_coarse, y_detail = ae(pcs)
#     print(v.size(), y_coarse.size(), y_detail.size())
