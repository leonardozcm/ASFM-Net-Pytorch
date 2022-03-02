import torch
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
from models.modelutils import fps_subsample
chamfer_dist = chamfer_3DDist()


def chamfer(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)


def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2


def chamfer_single_side(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(d1)
    return d1


def chamfer_single_side_sqrt(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(torch.sqrt(d1))
    return d1


def rmse_loss(yhat, y):
    loss_feat = torch.sqrt(
        torch.sum(torch.pow(torch.subtract(yhat, y), 2), 1, keepdim=False))
    loss_feat = torch.mean(loss_feat, dim=0, keepdim=False)
    return loss_feat

#
# def getLossAll(c1, c2, syn, coarse, fine, gt, alphas, step):
#     """
#     alpha:[
#         list[(step1, lr1),(step2,lr2)...],
#         ...
#         ]
#     step:int step
#     """
#     loss_feat = rmse_loss(c1, c2)
#
#     coarse_gt = gt
#     if coarse.shape[1] != gt.shape[1]:
#         coarse_gt = fps_subsample(gt, coarse.shape[1])
#         # print(coarse_gt.size())
#     loss_syn = chamfer_sqrt(syn, gt)
#     loss_coarse = chamfer_sqrt(coarse, coarse_gt)
#     loss_fine = chamfer_sqrt(fine, gt)
#
#     # determines alphas
#     def getalpha(schedule, step):
#         alpha = 0.0
#         for (point, alpha_) in schedule:
#             if step >= point:
#                 alpha = alpha_
#             else:
#                 break
#         return alpha
#
#     alphas_0 = getalpha(alphas[0], step)
#     alphas_1 = getalpha(alphas[1], step)
#     alphas_2 = getalpha(alphas[2], step)
#     # print(alphas_0, " ", alphas_1, " ", alphas_2)
#
#     loss_all = alphas_0 * loss_feat+alphas_1*loss_coarse+alphas_2*loss_fine
#     losses = [loss_feat, loss_coarse, loss_fine, loss_syn]
#     return loss_all, losses


def get_loss_finetune(pcds_pred, partial, gt, alphas=None, step=0, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    Pc, P1, P2, P3 = pcds_pred  # 4096, 512, 2048, 16384

    gt_2 = fps_subsample(gt, P2.shape[1])
    gt_1 = fps_subsample(gt_2, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    # print(Pc.size(), gt_c.size())
    cdc = CD(Pc, gt_c)
    # print(P1.size(), gt_1.size())
    cd1 = CD(P1, gt_1)
    # print(P2.size(), gt_2.size())
    cd2 = CD(P2, gt_2)
    # print(P3.size(), gt.size())
    cd3 = CD(P3, gt)

    partial_matching = PM(partial, P3)
    losses = [cdc, cd1, cd2, cd3, partial_matching]
    if alphas == None:
        return losses

    # determines alphas
    def getalpha(schedule, step):
        alpha = 0.0
        for (point, alpha_) in schedule:
            if step >= point:
                alpha = alpha_
            else:
                break
        return alpha

    alphas_0 = getalpha(alphas[0], step)
    alphas_1 = getalpha(alphas[1], step)
    # print(alphas_0, " ", alphas_1, " ", alphas_2)

    loss_all = ((cdc + cd1 + cd2 + partial_matching)
                * alphas_0 + cd3 * alphas_1) * 1e3
    losses = [cdc, cd1, cd2, cd3, partial_matching]
    return loss_all, losses


def get_loss(pcds_pred, partial, gt, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    Pc, P1, P2, P3 = pcds_pred

    gt_2 = fps_subsample(gt, P2.shape[1])
    gt_1 = fps_subsample(gt_2, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    cdc = CD(Pc, gt_c)
    # print(Pc.size(), gt_c.size())
    cd1 = CD(P1, gt_1)
    # print(P1.size(), gt_1.size())
    cd2 = CD(P2, gt_2)
    # print(P2.size(), gt_2.size())
    cd3 = CD(P3, gt)
    # print(P3.size(), gt.size())

    partial_matching = PM(partial, P3)

    loss_all = (cdc + cd1 + cd2 + cd3 + partial_matching) * 1e3
    losses = [cdc, cd1, cd2, cd3, partial_matching]
    return loss_all, losses
