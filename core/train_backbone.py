# -*- coding: utf-8 -*-
# @Author: XP

import logging
import os
import torch
from core.test_backbone import test_backbone
import utils.data_loaders
import utils.helpers
from datetime import datetime
from tqdm import tqdm
from time import time
from tensorboardX import SummaryWriter
from utils.average_meter import AverageMeter
from torch.optim.lr_scheduler import StepLR
from utils.schedular import GradualWarmupScheduler
from snowmodels.model import FeatureExtractor as Encoder
from models.pcn import Decoder
from models.modelutils import fps_subsample

from utils.loss_utils import rmse_loss


def SAModulesInit(path):
    """
    Args
        path: string, path to the model we trained in step 1.

    returns
        BL-Encoder:baseline encoder
        as_encoder: random init
        as_decoder: weighted
    """
    bl_encoder = Encoder()
    as_encoder = Encoder()
    as_decoder = Decoder()

    checkpoint = torch.load(path)
    update_dict = {}
    for k, v in checkpoint['model'].items():
        if k.startswith("module.encoder"):
            k_ = '.'.join(k.split('.')[2:])
            update_dict[k_] = v

    print(bl_encoder.load_state_dict(update_dict))
    for param in bl_encoder.parameters():
        param.requires_grad = False

    print("="*20)

    update_dict = {}
    for k, v in checkpoint['model'].items():
        if k.startswith("module.decoder"):
            k_ = '.'.join(k.split('.')[2:])
            update_dict[k_] = v

    print(as_decoder.load_state_dict(update_dict, strict=False))

    logging.info('AS Preparation Done!')
    return bl_encoder, as_encoder, as_decoder


def freezeDecoder(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreezeDecoder(model):
    for param in model.module.decoder.parameters():
        param.requires_grad = True


def train_backbone(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](
        cfg)
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](
        cfg)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TRAIN),
        batch_size=cfg.TRAIN.BACKBONE_BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKERS,
        collate_fn=utils.data_loaders.collate_fn,
        pin_memory=True,
        shuffle=True,
        drop_last=False)
    val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST),
        batch_size=cfg.TRAIN.BACKBONE_BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKERS//2,
        collate_fn=utils.data_loaders.collate_fn,
        pin_memory=True,
        shuffle=False)

    # Set up folders for logs and checkpoints
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', '%s',
                              datetime.now().isoformat())
    print(output_dir)
    cfg.DIR.CHECKPOINTS = output_dir % ('backbone', 'checkpoints')
    cfg.DIR.LOGS = output_dir % ('backbone', 'logs')
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    bl_encoder, as_encoder, as_decoder = SAModulesInit(cfg.CONST.PCNWEIGHTS)
    if torch.cuda.is_available():
        bl_encoder = torch.nn.DataParallel(bl_encoder).cuda()
        as_encoder = torch.nn.DataParallel(as_encoder).cuda()
        as_decoder = torch.nn.DataParallel(as_decoder).cuda()

    # Create the optimizers
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, as_encoder.parameters()),
                                 lr=cfg.TRAIN.BACKBONE_LEARNING_RATE,
                                 weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                 betas=cfg.TRAIN.BETAS)

    # lr scheduler
    lr_scheduler = StepLR(
        optimizer, step_size=50, gamma=0.7)

    accumulation_steps = 32 // cfg.TRAIN.BASELINE_BATCH_SIZE
    init_epoch = 0
    best_metrics = float('inf')
    steps = 0

    if 'BBWEIGHTS' in cfg.CONST:
        print("BBWEIGHTS FOUND")
        logging.info('Recovering from %s ...' % (cfg.CONST.BBWEIGHTS))
        checkpoint = torch.load(cfg.CONST.BBWEIGHTS)
        init_epoch = checkpoint["epoch_index"]
        best_metrics = checkpoint['best_metrics']
        steps = checkpoint['steps']
        as_encoder.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        logging.info('Recover complete. Current epoch = #%d; best metrics = %s.' % (
            init_epoch, best_metrics))
    else:
        print("BBWEIGHTS NOT FOUND")
        # no need 2 warmup
        lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
                                              after_scheduler=lr_scheduler)

    # Freeze the decoder of SA-module
    freezeDecoder(as_decoder)

    total_step = 0
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        count = 0

        epoch_start_time = time()

        batch_time = AverageMeter()

        as_encoder.train()

        total_cd_feat = 0

        batch_end_time = time()
        n_batches = len(train_data_loader)

        # 2 * bs(16) = 32(bs in paper)

        with tqdm(train_data_loader) as t:
            for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(t):

                # Debug switch
                # count += 1
                if count > 3:
                    break

                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)
                partial = data['partial_cloud']
                gt = data['gtcloud']

                #  downsample gt to 4096
                fine_gt = fps_subsample(gt, cfg.NETWORK.NUM_GT_POINTS)

                # preprocess transpose
                partial = partial.permute(0, 2, 1).contiguous()
                bl_inputs = fine_gt.permute(0, 2, 1).contiguous()

                v = as_encoder(partial)

                # feature matching loss
                v_complete = bl_encoder(bl_inputs)

                loss_feat = rmse_loss(v, v_complete)

                loss_feat = loss_feat / accumulation_steps
                loss_feat.backward()

                total_step += 1

                if (batch_idx+1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    cd_feat = loss_feat.item()
                    total_cd_feat += cd_feat

                    n_itr = (epoch_idx - 1) * n_batches + batch_idx
                    train_writer.add_scalar(
                        'Loss/Batch/feat_matching', loss_feat, n_itr)

                    batch_time.update(time() - batch_end_time)
                    batch_end_time = time()
                    t.set_description('[Epoch %d/%d][Batch %d/%d]' %
                                      (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches))
                    t.set_postfix(loss='%s' % ['%.4f' % l for l in [
                        cd_feat]])

                    if 'BBWEIGHTS' not in cfg.CONST and steps <= cfg.TRAIN.WARMUP_STEPS:
                        lr_scheduler.step()
                        steps += 1

        avg_feat = total_cd_feat / n_batches

        lr_scheduler.step()
        print('epoch: ', epoch_idx, 'optimizer: ',
              optimizer.param_groups[0]['lr'])
        epoch_end_time = time()

        train_writer.add_scalar(
            'Loss/Epoch/feat_matching', avg_feat, epoch_idx)
        logging.info(
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_feat]]))

        # Validate the current model
        cd_eval = test_backbone(
            cfg, epoch_idx, val_data_loader, val_writer, as_encoder, as_decoder, steps=total_step)

        # Save checkpoints
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or cd_eval < best_metrics:
            file_name = 'pcn-backbone-best.pth' if cd_eval < best_metrics else 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)

            lr_scheduler_save = lr_scheduler
            if 'BBWEIGHTS' not in cfg.CONST:  # train from scratch
                lr_scheduler_save = lr_scheduler_save.after_scheduler

            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': best_metrics,
                'steps': steps,
                'model': as_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler_save.state_dict()
            }, output_path)

            # for back up
            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': best_metrics,
                'steps': steps,
                'model': as_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler_save.state_dict()
            }, "./checkpoint/pcn-backbone-best.pth")

            logging.info('Saved checkpoint to %s ...' % output_path)
            if cd_eval < best_metrics:
                best_metrics = cd_eval

    train_writer.close()
    val_writer.close()
