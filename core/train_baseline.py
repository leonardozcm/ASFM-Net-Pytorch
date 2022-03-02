# -*- coding: utf-8 -*-
# @Author: XP

import logging
import os
import torch
from core.test_baseline import test_baseline
import utils.data_loaders
import utils.helpers
from datetime import datetime
from tqdm import tqdm
from time import time
from tensorboardX import SummaryWriter
from utils.average_meter import AverageMeter
from torch.optim.lr_scheduler import StepLR
from utils.schedular import GradualWarmupScheduler
from models.SApcn import ASFM
from utils.loss_utils import get_loss


def train_baseline(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](
        cfg)
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](
        cfg)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TRAIN),
        batch_size=cfg.TRAIN.BASELINE_BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKERS,
        collate_fn=utils.data_loaders.collate_fn,
        pin_memory=True,
        shuffle=True,
        drop_last=False)
    val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST),
        batch_size=cfg.TRAIN.BASELINE_BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKERS//2,
        collate_fn=utils.data_loaders.collate_fn,
        pin_memory=True,
        shuffle=False)

    # Set up folders for logs and checkpoints
    time_stamp = datetime.now().strftime('%m_%d_%H_%M_%S')
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', '%s',
                              time_stamp)
    print(output_dir)
    cfg.DIR.CHECKPOINTS = output_dir % ('baseline', 'checkpoints')
    cfg.DIR.LOGS = output_dir % ('baseline', 'logs')
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    model = ASFM()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # Create the optimizers
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=cfg.TRAIN.BACKBONE_LEARNING_RATE,
                                 weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                 betas=cfg.TRAIN.BETAS)

    # lr scheduler
    lr_scheduler = StepLR(
        optimizer, step_size=50, gamma=0.7)
    init_epoch = 0
    best_metrics = float('inf')
    steps = 0

    if 'PCNWEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.PCNWEIGHTS))
        checkpoint = torch.load(cfg.CONST.PCNWEIGHTS)
        init_epoch = checkpoint["epoch_index"]
        best_metrics = checkpoint['best_metrics']
        steps = checkpoint['steps']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        logging.info('Recover complete. Current epoch = #%d; best metrics = %s.' % (
            init_epoch, best_metrics))
    else:
        # no need 2 warmup
        lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
                                              after_scheduler=lr_scheduler)
    # Training/Testing the network
    count = 0
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):

        epoch_start_time = time()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        model.train()

        total_cd_pc = 0
        total_cd_p1 = 0
        total_cd_p2 = 0
        total_cd_p3 = 0
        total_partial = 0

        batch_end_time = time()
        n_batches = len(train_data_loader)

        accumulation_steps = 8  # 8 * bs(8) = 64(bs in paper)
        with tqdm(train_data_loader) as t:
            for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(t):
                # count += 1
                if count > 2:
                    break
                data_time.update(time() - batch_end_time)
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)
                partial = data['partial_cloud']
                gt = data['gtcloud']

                _, pcds_pred = model(partial=partial, gt=gt)

                loss_total, losses = get_loss(
                    pcds_pred, partial, gt, sqrt=True)

                loss_total = loss_total / accumulation_steps
                loss_total.backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    cd_pc_item = losses[0].item() * 1e3
                    total_cd_pc += cd_pc_item
                    cd_p1_item = losses[1].item() * 1e3
                    total_cd_p1 += cd_p1_item
                    cd_p2_item = losses[2].item() * 1e3
                    total_cd_p2 += cd_p2_item
                    cd_p3_item = losses[3].item() * 1e3
                    total_cd_p3 += cd_p3_item
                    partial_item = losses[4].item() * 1e3
                    total_partial += partial_item
                    n_itr = (epoch_idx - 1) * n_batches + batch_idx
                    train_writer.add_scalar(
                        'Loss/Batch/cd_pc', cd_pc_item, n_itr)
                    train_writer.add_scalar(
                        'Loss/Batch/cd_p1', cd_p1_item, n_itr)
                    train_writer.add_scalar(
                        'Loss/Batch/cd_p2', cd_p2_item, n_itr)
                    train_writer.add_scalar(
                        'Loss/Batch/cd_p3', cd_p3_item, n_itr)
                    train_writer.add_scalar(
                        'Loss/Batch/partial_matching', partial_item, n_itr)
                    batch_time.update(time() - batch_end_time)
                    batch_end_time = time()
                    t.set_description(
                        '[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches))
                    t.set_postfix(loss='%s' % ['%.4f' % l for l in
                                               [cd_pc_item, cd_p1_item, cd_p2_item, cd_p3_item, partial_item]])

                    if steps <= cfg.TRAIN.WARMUP_STEPS:
                        lr_scheduler.step()
                        steps += 1

        avg_cdc = total_cd_pc / n_batches
        avg_cd1 = total_cd_p1 / n_batches
        avg_cd2 = total_cd_p2 / n_batches
        avg_cd3 = total_cd_p3 / n_batches
        avg_partial = total_partial / n_batches

        lr_scheduler.step()
        print('epoch: ', epoch_idx, 'optimizer: ',
              optimizer.param_groups[0]['lr'])
        epoch_end_time = time()
        train_writer.add_scalar('Loss/Epoch/cd_pc', avg_cdc, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p1', avg_cd1, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p2', avg_cd2, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p3', avg_cd3, epoch_idx)
        train_writer.add_scalar(
            'Loss/Epoch/partial_matching', avg_partial, epoch_idx)
        logging.info(
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time,
             ['%.4f' % l for l in [avg_cdc, avg_cd1, avg_cd2, avg_cd3, avg_partial]]))

        # Validate the current model
        cd_eval = test_baseline(
            cfg, epoch_idx, val_data_loader, val_writer, model)

        # Save checkpoints
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or cd_eval < best_metrics:
            file_name = 'pcn-baseline-best.pth' if cd_eval < best_metrics else 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)

            lr_scheduler_save = lr_scheduler
            if 'PCNWEIGHTS' not in cfg.CONST:  # train from scratch
                lr_scheduler_save = lr_scheduler_save.after_scheduler

            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': best_metrics,
                'steps': steps,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler_save.state_dict()
            }, output_path)

            # for back up
            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': best_metrics,
                'steps': steps,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler_save.state_dict()
            }, "./checkpoint/pcn-baseline-best.pth")

            logging.info('Saved checkpoint to %s ...' % output_path)
            if cd_eval < best_metrics:
                best_metrics = cd_eval

    train_writer.close()
    val_writer.close()
