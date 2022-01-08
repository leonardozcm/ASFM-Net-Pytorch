import torch
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
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


def getLossAll(c1, c2, coarse, fine, gt, alphas, step):
    """
    alpha:[
        list[(step1, lr1),(step2,lr2)...],
        ...
        ]
    step:int step
    """
    loss_feat = rmse_loss(c1, c2)
    loss_coarse = chamfer_sqrt(coarse, gt)
    loss_fine = chamfer_sqrt(fine, gt)

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
    alphas_2 = getalpha(alphas[2], step)
    # print(alphas_0, " ", alphas_1, " ", alphas_2)

    loss_all = alphas_0 * loss_feat+alphas_1*loss_coarse+alphas_2*loss_fine
    losses = [loss_feat, loss_coarse, loss_fine]
    return loss_all, losses
