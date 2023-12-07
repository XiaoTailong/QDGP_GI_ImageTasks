#!/usr/bin/env python
# -*- coding: utf-8 -*-

import faulthandler
faulthandler.enable()
import os
import sys

# from collections import OrderedDict
import tensorflow as tf

import torch
import torchvision.utils as vutils


import utils
from models import QDGP_64_0001
# from torchsummary import summary

sys.path.append("./")
from str2bool import str2bool


## QNN啥都不需要输入，输入是分层噪声与类别标签
def add_qdgp_parser(parser):
    parser.add_argument(
        '--dist', action='store_true', default=False,
        help='Train with distributed implementation (default: %(default)s)')
    parser.add_argument(
        '--exp_path', type=str, default='',
        help='Experiment path (default: %(default)s)')
    parser.add_argument(
        '--root_dir', type=str, default='',
        help='Root path of dataset (default: %(default)s)')
    parser.add_argument(
        '--list_file', type=str, default='',
        help='List file of the dataset (default: %(default)s)')
    parser.add_argument(
        '--resolution', type=int, default=128,
        help='Resolution to resize the input image (default: %(default)s)')
    parser.add_argument(
        '--dgp_mode', type=str, default='reconstruct',
        help='DGP mode (default: %(default)s)')
    parser.add_argument(
        '--random_G', action='store_true', default=False,
        help='Use randomly initialized generator? (default: %(default)s)')
    parser.add_argument(
        '--update_G', action='store_true', default=True,
        help='Finetune Generator? (default: %(default)s)')
    parser.add_argument(
        '--update_embed', action='store_true', default=True,
        help='Finetune class embedding? (default: %(default)s)')
    parser.add_argument(
        '--save_G', action='store_true', default=False,
        help='Save fine-tuned generator and latent vector? (default: %(default)s)')
    parser.add_argument(
        '--print_interval', type=int, default=500, nargs='+',
        help='Number of iterations to print training loss (default: %(default)s)')
    parser.add_argument(
        '--save_interval', type=int, default=None, nargs='+',
        help='Number of iterations to save image')
    parser.add_argument(
        '--lr_ratio', type=float, default=[1.0, 1.0, 1.0, 1.0], nargs='+',
        help='Decreasing ratio for learning rate in blocks (default: %(default)s)')
    parser.add_argument(
        '--select_num', type=int, default=1000,
        help='Number of image pool to select from (default: %(default)s)')
    parser.add_argument(
        '--iterations', type=int, default=[2000, 1000, 1000, 1000], nargs='+',
        help='Training iterations for all stages')
    parser.add_argument(
        '--G_lrs', type=float, default=[1e-5, 5e-5, 1e-5, 1e-5], nargs='+',
        help='Learning rate steps of Generator')
    parser.add_argument(
        '--sample_std', type=float, default=0.3,
        help='sampling standard deviation')
    parser.add_argument(
        '--z_lrs', type=float, default=[1e-3, 1e-3, 1e-3, 1e-4], nargs='+',
        help='Learning rate steps of latent code z')
    parser.add_argument(
        '--warm_up', type=int, default=0,
        help='Number of warmup iterations (default: %(default)s)')
    parser.add_argument(
        '--use_in', type=str2bool, default=[True, True, True, True], nargs='+',
        help='Whether to use instance normalization in generator')

    return parser



# Arguments for demo
def add_example_parser(parser):
    parser.add_argument(
        '--image_path', type=str, default='',
        help='Path of the image to be processed (default: %(default)s)')
    parser.add_argument(
        '--bucket_path', type=str, default='',
        help='Path of the experimental bucket to be processed (default: %(default)s)'
    )
    parser.add_argument(
        '--pattern_path', type=str, default='',
        help='Path of the patterns to be processed (default: %(default)s)'
    )
    parser.add_argument(
        '--class', type=int, default=-1,
        help='class index of the image (default: %(default)s)')
    parser.add_argument('--dims', type=int, default=64, metavar='D',
                        help="dimension of the bucket data")
    parser.add_argument('--object', type=str, default="2",
                        help="image object")
    parser.add_argument('--measurement_setting', type=str, default="o",
                        help="image object")
    parser.add_argument('--n_qubits', type=int, default=20, help='number of qubits')
    parser.add_argument('--n_qlayers', type=int, default=4, help='number of layers')
    parser.add_argument('--n_heads', type=int, default=3, help='number of heads')

    return parser

# prepare arguments and save in config
parser = utils.prepare_parser()
parser = add_qdgp_parser(parser)
parser = add_example_parser(parser)
config = vars(parser.parse_args())
utils.dgp_update_config(config)


print("setting:", config['measurement_setting'], flush=True)

print("random_G:", config['random_G'], flush=True)

# set random seed
utils.seed_rng(config['seed'])

source_dir = os.path.dirname(os.path.abspath(__file__))
config['exp_path'] = source_dir
# config['exp_path'] = '/home/xtl/Documents/PythonFiles/SPI_QML/untrain_QuGe/deep-generative-prior'
#
# if not os.path.exists('{}/images'.format(config['exp_path'])):
#     os.makedirs('{}/images'.format(config['exp_path']))
# if not os.path.exists('{}/images_sheet'.format(config['exp_path'])):
#     os.makedirs('{}/images_sheet'.format(config['exp_path']))
# initialize DGP model
# print(qdgp.G, flush=True)  # good, no problem
# target image path (original)

config['image_path'] = source_dir + '/data_zhai/shot/{}.png'.format(config['object'])

config['bucket_path'] = source_dir + '/data_zhai/bucket_randomP/64/{}.npy'.format(config['object'])
config['pattern_path'] = source_dir + '/data_zhai/randomP/randomP.npy'
# prepare the target image
img = utils.get_img(config['image_path'], config['resolution']).cuda()
bucket_target = torch.Tensor(utils.get_bucket(config['bucket_path'], config['dims'])).cuda()
patterns = torch.Tensor(utils.get_pattern(config['pattern_path'], config['dims'])).cuda()
category = torch.Tensor([config['class']]).long().cuda()  # 默认是-1

## 测试原图是否可以重建

import numpy as np
def test_code(orignal_image, patterns, data_array):
    orignal_image = torch.Tensor(orignal_image).cuda()
    # patterns = torch.Tensor(patterns, dtype=torch.float32)  # with shape (1024*3, 64, 64)
    I_orginal = torch.Tensor([torch.sum(torch.multiply(orignal_image, patt)) for patt in patterns]).cuda()
    # I_orginal = torch.unsqueeze(I_orginal, dim=1)
    print(I_orginal, flush=True)
    print(torch.squeeze(data_array), flush=True)

    test_result = I_orginal - torch.Tensor(np.squeeze(data_array)).cuda()

    print(test_result, flush=True)

# import cv2
# orignal = cv2.imread(config['image_path'], 0)/255.0
#
# test_code(orignal, patterns, bucket_target)
# initialize DGP model
qdgp = QDGP_64_0001(config)

# qdgp.set_target(bucket_target, category)
# we need to ensure the shapes of the tensor
# prepare initial latent vector
# qdgp.select_quantum_z(patterns, select_y=True if config['class'] < 0 else False)
# start reconstruction
loss_dict = qdgp.run(patterns, bucket_target)  # 这一步已经优化完成了


# save_imgs = img.clone().cpu()



