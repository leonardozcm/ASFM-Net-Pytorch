# -*- coding: utf-8 -*-
# @Author: XP

from easydict import EasyDict as edict

__C = edict()
cfg = __C

#
# Dataset Config
#
__C.DATASETS = edict()
__C.DATASETS.COMPLETION3D = edict()
__C.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH = './datasets/Completion3D.json'
__C.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH = '/path/to/datasets/Completion3D/%s/partial/%s/%s.h5'
__C.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH = '/path/to/datasets/Completion3D/%s/gt/%s/%s.h5'
__C.DATASETS.SHAPENET = edict()
__C.DATASETS.SHAPENET.CATEGORY_FILE_PATH = './datasets/ShapeNet.json'
__C.DATASETS.SHAPENET.N_RENDERINGS = 8
__C.DATASETS.SHAPENET.N_POINTS = 2048
__C.DATASETS.SHAPENET.PARTIAL_POINTS_PATH = '/home/chriskafka/dataset/ShapeNetCompletion/%s/partial/%s/%s/%02d.pcd'
__C.DATASETS.SHAPENET.COMPLETE_POINTS_PATH = '/home/chriskafka/dataset/ShapeNetCompletion/%s/complete/%s/%s.pcd'

#
# Dataset
#
__C.DATASET = edict()
# Dataset Options: Completion3D, ShapeNet, ShapeNetCars, Completion3DPCCT
__C.DATASET.TRAIN_DATASET = 'ShapeNet'
__C.DATASET.TEST_DATASET = 'ShapeNet'

#
# Constants
#
__C.CONST = edict()

__C.CONST.NUM_WORKERS = 8
__C.CONST.N_INPUT_POINTS = 2048

#
# Directories
#

__C.DIR = edict()
__C.DIR.OUT_PATH = './checkpoint'
__C.CONST.DEVICE = '0, 1, 2, 3'
# __C.CONST.WEIGHTS = 'checkpoint/baseline/checkpoints/2021-12-31T15:16:37.450752/pcnbackbone-best.pth'
# __C.CONST.PCNWEIGHTS = 'checkpoint/baseline/checkpoints/2021-12-31T15:16:37.450752/pcnbackbone-best.pth'
# __C.CONST.BBWEIGHTS = 'checkpoint/baseline/checkpoints/2021-12-31T15:16:37.450752/pcnbackbone-best.pth'

#
# Memcached
#
__C.MEMCACHED = edict()
__C.MEMCACHED.ENABLED = False
__C.MEMCACHED.LIBRARY_PATH = '/mnt/lustre/share/pymc/py3'
__C.MEMCACHED.SERVER_CONFIG = '/mnt/lustre/share/memcached_client/server_list.conf'
__C.MEMCACHED.CLIENT_CONFIG = '/mnt/lustre/share/memcached_client/client.conf'

#
# Network
#
__C.NETWORK = edict()
__C.NETWORK.N_SAMPLING_POINTS = 2048
__C.NETWORK.NUM_GT_POINTS = 4096

#
# Train
#
__C.TRAIN = edict()

__C.TRAIN.BASELINE_BATCH_SIZE = 8
__C.TRAIN.BACKBONE_BATCH_SIZE = 32


__C.TRAIN.N_EPOCHS = 800
__C.TRAIN.SAVE_FREQ = 25
__C.TRAIN.LEARNING_RATE = 0.001
__C.TRAIN.BACKBONE_LEARNING_RATE = 0.0001
__C.TRAIN.LR_MILESTONES = [50, 100, 150, 200, 250]
__C.TRAIN.LR_DECAY_STEP = 50
__C.TRAIN.WARMUP_STEPS = 200
__C.TRAIN.GAMMA = .5
__C.TRAIN.BETAS = (.9, .999)
__C.TRAIN.WEIGHT_DECAY = 0

#
# Test
#
__C.TEST = edict()
__C.TEST.METRIC_NAME = 'ChamferDistance'
