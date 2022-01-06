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
from utils.loss_utils import chamfer_sqrt
from models.pcn import AutoEncoder
from models.utils import fps_subsample


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
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', '%s',
                              datetime.now().isoformat())
    print(output_dir)
    cfg.DIR.CHECKPOINTS = output_dir % ('baseline', 'checkpoints')
    cfg.DIR.LOGS = output_dir % ('baseline', 'logs')
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    model = AutoEncoder()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # Create the optimizers
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=cfg.TRAIN.BACKBONE_LEARNING_RATE,
                                 weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                 betas=cfg.TRAIN.BETAS)

    # lr scheduler
    lr_scheduler = StepLR(
        optimizer, step_size=20, gamma=0.7)
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

        total_cd_fine = 0

        batch_end_time = time()
        n_batches = len(train_data_loader)

        # 4 * bs(8) = 32(bs in paper)
        accumulation_steps = 32 // cfg.TRAIN.BASELINE_BATCH_SIZE
        with tqdm(train_data_loader) as t:
            for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(t):

                # Debug switch
                # count += 1
                if count > 2:
                    break

                data_time.update(time() - batch_end_time)
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)
                # partial = data['partial_cloud']
                gt = data['gtcloud']

                #  downsample gt to 4096
                gt = fps_subsample(gt, cfg.NETWORK.NUM_GT_POINTS)
                input_pl = gt

                # preprocess transpose
                input_pl = input_pl.permute(0, 2, 1)

                v, _, y_detail = model(input_pl)

                y_detail = y_detail.permute(0, 2, 1)

                loss_fine = chamfer_sqrt(gt, y_detail)
                # print(gt.shape, " ", y_detail.shape)
                loss = loss_fine
                loss = loss / accumulation_steps
                loss.backward()

                if (batch_idx+1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    cd_fine = loss_fine.item() * 1e3
                    total_cd_fine += cd_fine

                    n_itr = (epoch_idx - 1) * n_batches + batch_idx

                    train_writer.add_scalar(
                        'Loss/Batch/cd_fine', cd_fine, n_itr)

                    batch_time.update(time() - batch_end_time)
                    batch_end_time = time()
                    t.set_description('[Epoch %d/%d][Batch %d/%d]' %
                                      (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches))
                    t.set_postfix(loss='%s' % ['%.4f' % l for l in [
                        cd_fine]])

                    if 'PCNWEIGHTS' not in cfg.CONST and steps <= cfg.TRAIN.WARMUP_STEPS:
                        lr_scheduler.step()
                        steps += 1

        avg_cdf = total_cd_fine / n_batches

        lr_scheduler.step()
        print('epoch: ', epoch_idx, 'optimizer: ',
              optimizer.param_groups[0]['lr'])
        epoch_end_time = time()
        train_writer.add_scalar('Loss/Epoch/cd_fine', avg_cdf, epoch_idx)
        logging.info(
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cdf]]))

        # Validate the current model
        cd_eval = test_baseline(
            cfg, epoch_idx, val_data_loader, val_writer, model)

        # Save checkpoints
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or cd_eval < best_metrics:
            file_name = 'pcnbackbone-best.pth' if cd_eval < best_metrics else 'ckpt-epoch-%03d.pth' % epoch_idx
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

            logging.info('Saved checkpoint to %s ...' % output_path)
            if cd_eval < best_metrics:
                best_metrics = cd_eval

    train_writer.close()
    val_writer.close()
