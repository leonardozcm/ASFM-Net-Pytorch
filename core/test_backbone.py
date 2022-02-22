# -*- coding: utf-8 -*-
# @Author: XP

import logging
import torch
from models.modelutils import fps_subsample
import utils.data_loaders
import utils.helpers
from tqdm import tqdm
from utils.average_meter import AverageMeter
from utils.metrics import Metrics
from utils.loss_utils import chamfer_sqrt
from models.SApcn import ASFM as Model


def test_backbone(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, model=None, steps=0):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    if test_data_loader is None:
        # Set up data loader
        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](
            cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetSubset.TEST),
            batch_size=1,
            num_workers=cfg.CONST.NUM_WORKERS,
            collate_fn=utils.data_loaders.collate_fn,
            pin_memory=True,
            shuffle=False)

    # Setup networks and initialize networks
    if model is None:
        model = Model(step_ratio=4)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()

        logging.info('Recovering from %s ...' % (cfg.CONST.BBWEIGHTS))
        checkpoint = torch.load(cfg.CONST.BBWEIGHTS)
        model.load_state_dict(checkpoint['model'])

    # Switch models to evaluation mode
    model.eval()

    n_samples = len(test_data_loader)
    test_losses = AverageMeter(
        ['cd_coarse', 'cd_fine', 'cd_syn'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()

    # Testing loop
    with tqdm(test_data_loader) as t:
        for model_idx, (taxonomy_id, model_id, data) in enumerate(t):
            taxonomy_id = taxonomy_id[0] if isinstance(
                taxonomy_id[0], str) else taxonomy_id[0].item()
            model_id = model_id[0]

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                partial = data['partial_cloud']
                gt = data['gtcloud']

                # downsample gt to 2048
                # partial = fps_subsample(gt, 2048)
                fine_gt = fps_subsample(gt, 4096)

                # preprocess transpose
                partial = partial.permute(0, 2, 1)

                v, arr_pcd = model(partial)
                # 2048, 4096
                y_syn, y_coarse, y_detail = arr_pcd[0], arr_pcd[2], arr_pcd[3]

                coarse_gt = gt
                if y_coarse.shape[1] != gt.shape[1]:
                    coarse_gt = fps_subsample(gt, y_coarse.shape[1])
                    # print(coarse_gt.size())  # 2048
                loss_coarse = chamfer_sqrt(y_coarse, coarse_gt)

                loss_fine = chamfer_sqrt(fine_gt, y_detail)

                loss_syn = chamfer_sqrt(fine_gt, y_syn)

                cd_coarse = loss_coarse.item() * 1e3

                cd_fine = loss_fine.item() * 1e3

                cd_syn = loss_syn.item() * 1e3

                _metrics = [loss_fine]
                test_losses.update([cd_coarse, cd_fine, cd_syn])

                test_metrics.update(_metrics)
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(
                        Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

                t.set_description('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                                  (model_idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()
                                                                                     ], ['%.4f' % m for m in _metrics]))

    # Print testing results
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for metric in test_metrics.items:
        print(metric, end='\t')
    print()

    for taxonomy_id in category_metrics:
        print(taxonomy_id, end='\t')
        print(category_metrics[taxonomy_id].count(0), end='\t')
        for value in category_metrics[taxonomy_id].avg():
            print('%.4f' % value, end='\t')
        print()

    print('Overall', end='\t\t\t')
    for value in test_metrics.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    print('Epoch ', epoch_idx, end='\t')
    for value in test_losses.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Loss/Epoch/cd_coarse',
                               test_losses.avg(0), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/cd_syn',
                               test_losses.avg(2), epoch_idx)
        if steps > cfg.TRAIN.STEP_STAGE_1 + 100:
            test_writer.add_scalar('Loss/Epoch/cd_fine',
                                   test_losses.avg(1), epoch_idx)
            for i, metric in enumerate(test_metrics.items):
                test_writer.add_scalar('Metric/%s' %
                                       metric, test_metrics.avg(i), epoch_idx)

    return test_losses.avg(1)
