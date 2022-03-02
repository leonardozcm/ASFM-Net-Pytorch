# -*- coding: utf-8 -*-
# @Author: XP

import logging
import os
import torch
from core.test_finetune import test_finetune
import utils.data_loaders
import utils.helpers
from datetime import datetime
from tqdm import tqdm
from time import time
from tensorboardX import SummaryWriter
from utils.average_meter import AverageMeter
from torch.optim.lr_scheduler import StepLR
from utils.loss_utils import get_loss_finetune
from models.SApcn import ASFM


def ModulesInit(path, up_factors=[4, 8], freeze=True):
    """
    Args
        path: string, path to the model we trained in step 1.
    returns
        BL-Encoder:baseline encoder
        SA-AutoEncoder
    """
    as_autoencoder = ASFM(up_factors=up_factors)

    encoder_checkpoint = torch.load(
        os.path.join(path, "pcn-backbone-best.pth"))
    update_dict = {}
    for k, v in encoder_checkpoint['model'].items():

        k_ = '.'.join(k.split('.')[1:])
        update_dict["encoder."+k_] = v

    decoder_checkpoint = torch.load(
        os.path.join(path, "pcn-baseline-best.pth"))
    for k, v in decoder_checkpoint['model'].items():
        if k.startswith("module.decoder"):
            k_ = '.'.join(k.split('.')[1:])
            update_dict[k_] = v

    if freeze:
        for param in as_autoencoder.encoder.parameters():
            param.requires_grad = False

        for param in as_autoencoder.decoder.parameters():
            param.requires_grad = False

    print(as_autoencoder.load_state_dict(update_dict, strict=False))

    logging.info('AS Preparation Done!')
    return as_autoencoder


def freezeDecoder(model):
    for param in model.module.decoder.parameters():
        param.requires_grad = False


def unfreezeDecoder(model):
    for param in model.module.decoder.parameters():
        param.requires_grad = True


def getAlphaSchedule(cfg=None):

    step_stages = [0,
                   50,
                   100,
                   150]

    # schedule = [[1., 1., 0.5, 0.],
    #             [0.5, 0.5, 1., 1.]]
    schedule = [[1., 1., 0.5, 0.],
                [1., 1., 1., 1.]]

    schedule_new = []
    for ls in schedule:
        ls_new = []
        for i, a in enumerate(ls):
            ls_new.append((step_stages[i], a))
        schedule_new.append(ls_new)
    return schedule_new


def train_finetune(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](
        cfg)
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](
        cfg)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TRAIN),
        batch_size=cfg.TRAIN.FINETUNE_BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKERS,
        collate_fn=utils.data_loaders.collate_fn,
        pin_memory=True,
        shuffle=True,
        drop_last=False)
    val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST),
        batch_size=cfg.TRAIN.FINETUNE_BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKERS//2,
        collate_fn=utils.data_loaders.collate_fn,
        pin_memory=True,
        shuffle=False)

    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', '%s',
                              datetime.now().isoformat())
    print(output_dir)
    cfg.DIR.CHECKPOINTS = output_dir % ('finetune', 'checkpoints')
    cfg.DIR.LOGS = output_dir % ('finetune', 'logs')
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    as_autoencoder = ModulesInit(cfg.DIR.OUT_PATH, freeze=False)
    if torch.cuda.is_available():
        as_autoencoder = torch.nn.DataParallel(as_autoencoder).cuda()

    # Create the optimizers
    optimizer = torch.optim.Adam(as_autoencoder.parameters(),
                                 lr=cfg.TRAIN.BACKBONE_LEARNING_RATE,
                                 weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                 betas=cfg.TRAIN.BETAS)

    # lr scheduler
    lr_scheduler = StepLR(
        optimizer, step_size=50, gamma=0.7)

    accumulation_steps = 64 // cfg.TRAIN.FINETUNE_BATCH_SIZE
    print("accumulation_steps:", accumulation_steps)
    alpha_schedules = getAlphaSchedule()
    init_epoch = 0
    best_metrics = float('inf')
    steps = 0

    if 'FTWEIGHTS' in cfg.CONST:
        print("FTWEIGHTS FOUND")
        logging.info('Recovering from %s ...' % (cfg.CONST.FTWEIGHTS))
        checkpoint = torch.load(cfg.CONST.FTWEIGHTS)
        init_epoch = checkpoint["epoch_index"]
        best_metrics = checkpoint['best_metrics']
        steps = checkpoint['steps']
        as_autoencoder.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        logging.info('Recover complete. Current epoch = #%d; best metrics = %s.' % (
            init_epoch, best_metrics))
    else:
        print("FTWEIGHTS NOT FOUND")
        # no need 2 warmup
        # lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
        #                                       after_scheduler=lr_scheduler)

    # Freeze the decoder of SA-module
    # freezeDecoder(as_autoencoder)

    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
       # unfreeze backbone
        if epoch_idx == 50:
            for param in as_autoencoder.encoder.parameters():
                param.requires_grad = True

            for param in as_autoencoder.decoder.parameters():
                param.requires_grad = True

        count = 0

        epoch_start_time = time()

        batch_time = AverageMeter()

        as_autoencoder.train()

        total_cd_pc = 0
        total_cd_p1 = 0
        total_cd_p2 = 0
        total_cd_p3 = 0
        total_partial = 0

        batch_end_time = time()
        n_batches = len(train_data_loader)

        # 2 * bs(16) = 32(bs in paper)

        with tqdm(train_data_loader) as t:
            for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(t):

                # Debug switch
                count += 1
                if count > 3:
                    break

                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)
                partial = data['partial_cloud']
                gt = data['gtcloud']

                _, arr_pcd = as_autoencoder(partial)

                loss, losses = get_loss_finetune(
                    arr_pcd, partial, gt, alpha_schedules, epoch_idx)

                loss = loss / accumulation_steps
                loss.backward()

                if (batch_idx+1) % accumulation_steps == 0:
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
                    t.set_description('[Epoch %d/%d][Batch %d/%d]' %
                                      (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches))
                    t.set_postfix(loss='%s' % ['%.4f' % l for l in
                                               [cd_pc_item, cd_p1_item, cd_p2_item, cd_p3_item, partial_item]])
                    if 'FTWEIGHTS' not in cfg.CONST and steps <= cfg.TRAIN.WARMUP_STEPS:
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
            (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cdc, avg_cd1, avg_cd2, avg_cd3, avg_partial]]))

        # Validate the current model
        cd_eval = test_finetune(
            cfg, epoch_idx, val_data_loader, val_writer, as_autoencoder)

        # Save checkpoints
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or cd_eval < best_metrics:
            file_name = 'pcn-finetune-best.pth' if cd_eval < best_metrics else 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)

            lr_scheduler_save = lr_scheduler
            # if 'FTWEIGHTS' not in cfg.CONST:  # train from scratch
            #     lr_scheduler_save = lr_scheduler_save.after_scheduler

            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': best_metrics,
                'steps': steps,
                'model': as_autoencoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler_save.state_dict()
            }, output_path)

            # for back up
            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': best_metrics,
                'steps': steps,
                'model': as_autoencoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler_save.state_dict()
            }, "./checkpoint/pcn-finetune-best.pth")

            logging.info('Saved checkpoint to %s ...' % output_path)
            if cd_eval < best_metrics:
                best_metrics = cd_eval

    train_writer.close()
    val_writer.close()
