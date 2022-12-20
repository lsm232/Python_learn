#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from datasets import *
from noise2noise import Noise2Noise
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parameters
    parser.add_argument('-t', '--train-dir', help='training set path', default='/data/liaoshimin/cycle_self/data/stage2/train/trainB')
    parser.add_argument('-v', '--valid-dir', help='test set path', default='/data/liaoshimin/cycle_self/data/stage2/test/')
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='/data/liaoshimin/cycle_self/s2/noise2noise-pytorch-master/checkpoints/')
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true')
    parser.add_argument('--report-interval', help='batch report interval', default=668, type=int)  #default 334
    parser.add_argument('-ts', '--train-size', help='size of train dataset', type=int,default=2672)  #这里指的不是训练集图像尺寸的大小，而是训练集的数量，这里之所以删除一张是因为 ts必须整除report_interval
    parser.add_argument('-vs', '--valid-size', help='size of valid dataset', type=int,default=712)

    # Training hyperparameters
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.0002, type=float)  #default 0.001
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=4, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=100, type=int)
    parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2', 'hdr'], default='l1', type=str)
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true')

    # Corruption parameters
    parser.add_argument('-n', '--noise-type', help='noise type',
        choices=['gaussian', 'poisson', 'text', 'mc'], default='gaussian', type=str)
    parser.add_argument('-p', '--noise-param', help='noise parameter (e.g. std for gaussian)', default=0.25, type=float)   #原本是50 ，参考论文CycleGAN denoising of extreme low-dose cardiac CT usingwavelet-assisted noise disentanglement
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='random crop size', default=256, type=int)
    parser.add_argument('--clean-targets', help='use clean targets for training', action='store_true')

    # lsm add
    parser.add_argument('--test_flag', default=0, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    """Trains Noise2Noise."""



    # Parse training parameters
    params = parse_args()

    # Train/valid datasets
    train_loader = load_dataset(params.train_dir, params.train_size, params, shuffled=True)
    #valid_loader = load_dataset(params.valid_dir, params.valid_size, params, shuffled=False)
    valid_loader = load_dataset_lsm(params.valid_dir+'/testA/',params.valid_dir+'/testB/',params)

    # Initialize model and train
    n2n = Noise2Noise(params, trainable=True)
    n2n.train(train_loader, valid_loader)
