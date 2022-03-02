#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Xiang

import argparse
import logging
import os
import numpy as np
import sys
import torch
from pprint import pprint
from config_pcn import cfg

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args_from_command_line():
    parser = argparse.ArgumentParser(
        description='The argument parser of SnowflakeNet')
    parser.add_argument('--test', dest='test',
                        help='Test neural networks', action='store_true')
    parser.add_argument('--baseline', dest='baseline',
                        help='First train the PCN as baseline', action='store_true')
    parser.add_argument('--backbone', dest='backbone',
                        help='Second train the Asymmetrical Siamese PCN as backbone', action='store_true')
    parser.add_argument('--finetune', dest='finetune',
                        help='Second train the Asymmetrical Siamese PCN as finetune', action='store_true')
    parser.add_argument('--inference', dest='inference',
                        help='Inference for benchmark', action='store_true')
    args = parser.parse_args()

    return args


def main():
    # Get args from command line
    args = get_args_from_command_line()
    print('cuda available ', torch.cuda.is_available())

    # Print config
    # print('Use config:')
    # pprint(cfg)

    if not args.test and not args.inference:
        if args.baseline:
            from core.train_baseline import train_baseline
            train_baseline(cfg)
        elif args.backbone:
            from core.train_backbone import train_backbone
            train_backbone(cfg)
        elif args.finetune:
            from core.train_finetune import train_finetune
            train_finetune(cfg)
        else:
            from core.train_pcn import train_net
            train_net(cfg)
    else:
        if cfg.CONST.BBWEIGHTS is None:
            raise Exception(
                'Please specify the path to checkpoint in the configuration file!')

        if args.test:
            if args.baseline:
                from core.test_baseline import test_baseline
                test_baseline(cfg)
            elif args.backbone:
                from core.test_backbone import test_backbone
                test_backbone(cfg)
            else:
                from core.test_pcn import test_net
                test_net(cfg)
        else:
            from core.inference_pcn import inference_net
            inference_net(cfg)


if __name__ == '__main__':
    # Check python version
    seed = 1
    set_seed(seed)
    # logging.basicConfig(
    #     format='[%(levelname)s] %(asctime)s %(message)s', level=logging.DEBUG)
    main()
