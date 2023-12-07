import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from PIL import Image
# from skimage import color
# from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import math
from models.QSampler import EnsembleQCBM, EnsembleQCBM_single
import models
import utils
from models.downsampler import Downsampler


## revise the code to define the QDGP model for pretraining
class QDGP_hybrid(nn.Module):
    def __init__(self, config):
        super(QDGP_hybrid, self).__init__()

        self.use_in = config['use_in']
        self.factor = 2 if config['resolution'] == 256 else 1  # 128 * 128 image resolution
        self.N_heads = config['n_heads']
        self.N_qubits = config['n_qubits']
        self.N_layers = config['n_qlayers']
        self.measurement_setting = config['measurement_setting']

        ## quantum NN
        if self.measurement_setting == 'd':
            self.QNNPrior = EnsembleQCBM(self.N_heads, self.N_qubits, self.N_layers, self.measurement_setting)
        elif self.measurement_setting == 's':
            self.QNNPrior = EnsembleQCBM_single(self.N_heads, self.N_qubits, self.N_layers, self.measurement_setting)
        # classical net
        self.G = models.Generator(**config)

        self.downsampler = Downsampler(
            n_planes=3,  # 3通道进行下采样，出来的结果也是3通道的
            factor=self.factor,
            kernel_type='lanczos2',
            phase=0.5,
            preserve_size=True).type(torch.cuda.FloatTensor)

    def forward(self, x, y, temperature):
        qout = self.QNNPrior(x, temperature)
        # print(qout, flush=True)  # 看起是否发生变化

        rec_image = self.G(qout, self.G.shared(y), use_in=self.use_in[0])
        # down sampler to 128 * 128 with single channel
        rec_image = self.downsampler((rec_image + 1) / 2)  # downsample to (1, 3, 64, 64)
        # rec_image = transforms.Grayscale()(rec_image)  # with shape (1, 1, 64, 64)

        return rec_image, qout


class QDGP(object):
    def __init__(self, config):
        # self.target_bucket = None
        self.rank, self.world_size = 0, 1
        if config['dist']:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        self.config = config
        self.update_G = config['update_G']
        self.update_embed = config['update_embed']
        self.iterations = config['iterations']
        # self.ftr_num = config['ftr_num']
        # self.ft_num = config['ft_num']
        self.lr_ratio = config['lr_ratio']  # learning rate ratio for generator and prior
        self.G_lrs = config['G_lrs']
        self.z_lrs = config['z_lrs']
        self.use_in = config['use_in']
        self.select_num = config['select_num']

        self.dim = config['dims']
        self.object = config['object']
        # create model
        if torch.cuda.is_available():
            self.model = QDGP_hybrid(config).cuda()
        else:
            self.model = QDGP_hybrid(**config)
        self.D = None  # 不需要NN判别器，有物理损失
        # self.model.G.optim = torch.optim.Adam(
        #     [{'params': self.model.G.get_params(i, self.update_embed)}
        #      for i in range(len(self.model.G.blocks) + 1)],
        #     lr=self.G_lrs[0],  # 采用learning rate 固定的方式 不进行learning rate自适应调节
        # )

        # load weights, we can choose different mode to compare the pretraing and non-pretraining
        if config['random_G']:
            self.random_G()
        else:
            utils.load_weights(
                self.model.G if not (config['use_ema']) else None,
                self.D,  # D is none and has no effect
                config['weights_root'],
                name_suffix=config['load_weights'],
                G_ema=self.model.G if config['use_ema'] else None,
                strict=False)

        self.model.G.eval()
        if self.D is not None:  # D is none
            self.D.eval()
        self.G_weight = deepcopy(self.model.G.state_dict())

        # prepare latent variable and optimizer
        # self._prepare_quantum_latent()
        # prepare learning rate scheduler
        # self.G_scheduler = utils.LRScheduler(self.G.optim, config['warm_up'])
        # self.z_scheduler = utils.LRScheduler(self.z_optim, config['warm_up'])

        self.criterion = utils.PhysicalLoss()

    # def _prepare_quantum_latent(self):
    #     # define the quantum sampler
    #     if torch.cuda.is_available():
    #         self.y = torch.zeros(1).long().cuda()
    #     else:
    #         self.y = torch.zeros(1).long()

    def reset_G(self):
        self.model.G.load_state_dict(self.G_weight, strict=False)
        self.model.G.reset_in_init()
        if self.config['random_G']:
            self.model.G.train()
        else:
            self.model.G.eval()

    def random_G(self):
        self.model.G.init_weights()

    # def set_target(self, target, category):
    #     self.target_bucket = target
    #     self.y.fill_(category.item())

    def run(self, patterns, bucket_target, save_interval=None):

        loss_dict = {}
        curr_step = 0

        # parameter_analysis(self.G)
        # temperature = 1.
        loss_list = []
        # there are two choices, one for zero input and one for random input
        if self.config['measurement_setting'] == 'd' or self.config['measurement_setting'] == 's':
            x = torch.rand((1, 120)) * 2 - 1
            x = x.cuda()
        else:
            x = torch.zeros(1, self.model.N_qubits).cuda()
        # random class label for G
        y = torch.randint(0, self.config['n_classes'], [1]).long().cuda()

        for stage, iteration in enumerate(self.iterations):
            # 分了不同的阶段，每个阶段都进行无训练学习
            self.QNN_optim = torch.optim.Adam(
                [{'params': self.model.QNNPrior.parameters(),
                  'lr': self.config['z_lrs'][stage]}],
                betas=(self.config['G_B1'], self.config['G_B2']),
                weight_decay=0,
                eps=1e-8
            )
            self.model.G.optim = torch.optim.Adam(
                [{'params': self.model.G.get_params(i, self.update_embed)}
                 for i in range(len(self.model.G.blocks) + 1)],
                lr=self.config['G_lrs'][stage]
                # 采用learning rate 固定的方式 不进行learning rate自适应调节
            )
            temperature = 1.0
            for i in range(iteration):
                start = time.time()
                # temperature = 0.0001 + (1 - 0.0001) * math.exp(-1. * i / 0.8)
                # temperature = temperature - (temperature - 0.0001) / iteration * i
                curr_step += 1
                # setup learning rate
                # self.G_scheduler.update(curr_step, self.G_lrs[stage])
                # self.z_scheduler.update(curr_step, self.z_lrs[stage])

                self.QNN_optim.zero_grad()

                if self.update_G:
                    self.model.G.optim.zero_grad()

                # use_in is whether using instance-norm
                # print_z = self.z.clone()
                # print(print_z.detach().cpu().numpy(), flush=True)
                rec_image, qout = self.model(x, y, temperature)
                _, _, width, height = rec_image.shape
                tv_loss = 1e-5 * total_variation_loss(image=rec_image)

                bucket = PhysicalForward(rec_image, patterns)
                # print(bucket.shape, flush=True)
                # print(self.target_bucket.shape, flush=True)
                bucket_loss = F.mse_loss(bucket, bucket_target.cuda()) / (width ** 2)
                # 加入了TV正则化
                loss = bucket_loss + tv_loss
                loss.backward()

                # if self.config['measurement_setting'] == 't':
                #     if i % 100 == 0:
                #         self.QNN_optim.step()
                # else:
                self.QNN_optim.step()

                if self.update_G:  # 实时更新G。
                    self.model.G.optim.step()

                loss_list.append(loss.detach().cpu().numpy())
                loss_dict = {'mse_loss': bucket_loss}
                end = time.time()
                if i == 0 or (i + 1) % self.config['print_interval'] == 0:
                    if self.rank == 0:
                        print(', '.join(
                            ['Stage: [{0}/{1}]'.format(stage + 1, len(self.iterations))] +
                            ['Iter: [{0}/{1}]'.format(i + 1, iteration)] +
                            ['%s : %+4.8f' % (key, loss_dict[key]) for key in loss_dict] +
                            ['Cost: {:+.8f}'.format(end - start)]
                        ), flush=True)
                        if self.config['random_G']:
                            np.save("./experiments/data_recon/{}/QDGP128_noise_mode{}_recons_dim{}_iter{}.npy"
                                    .format(self.object, self.model.measurement_setting, self.dim, curr_step),
                                    np.array(loss_list))
                            self.to_img(rec_image.squeeze().detach().cpu().numpy(), curr_step, True)
                            np.save("./experiments/data_recon/{}/QDGP128_noise_mode{}_prior_dim{}_iter{}.npy"
                                    .format(self.object, self.model.measurement_setting, self.dim, curr_step),
                                    qout.detach().cpu().numpy())
                            np.save("./experiments/data_recon/{}/QDGP128_noise_mode{}_prior_dim{}_iter{}.npy".format(
                                self.object, self.model.measurement_setting, self.dim, curr_step),
                                qout.detach().cpu().numpy())
                        else:
                            np.save("./experiments/data_recon/{}/QDGP128_noise_pretrain_mode{}_recons_dim{}_iter{}.npy"
                                    .format(self.object, self.model.measurement_setting, self.dim,
                                            curr_step),
                                    qout.detach().cpu().numpy())
                            np.save("./experiments/data_recon/{}/QDGP128_noise_pretrain_mode{}_prior_dim{}_iter{}.npy".format(
                                self.object, self.model.measurement_setting, self.dim, curr_step),
                                qout.detach().cpu().numpy())
                            self.to_img(rec_image.squeeze().detach().cpu().numpy(), curr_step, False)

        return loss_dict

    def to_img(self, images, iterations, random_G):
        # images = np.squeeze(images)
        images = (images - np.min(images)) / (np.max(images) - np.min(images))
        images = images * 255.0
        im = Image.fromarray(images)
        im = im.convert("L")
        if random_G:
            im.save("./experiments/data_recon/{}/QDGP128_noise_mode{}_recons_dim{}_iter{}.jpeg"
                    .format(self.object,
                   self.model.measurement_setting,
                   self.dim,
                   iterations))
        else:
            im.save("./experiments/data_recon/{}/QDGP128_noise_pretrain_mode{}_recons_dim{}_iter{}.jpeg"
                    .format(self.object,
                            self.model.measurement_setting,
                            self.dim,
                            iterations))


# Physical forward process
def PhysicalForward(image, patterns):
    # if not torch.is_tensor(patterns):
    #     patterns = torch.Tensor(patterns, dtype=torch.float32)
    image = torch.squeeze(image)  # 去掉batch dimension与channel维度
    bucket = torch.sum(torch.sum(torch.multiply(patterns, image), dim=-1), dim=-1)
    bucket = torch.unsqueeze(bucket, dim=1)
    return bucket


def total_variation_loss(image):
    # Calculate the total variation loss for the given image
    # image: Tensor of shape (batch_size, channels, height, width)

    batch_size, channels, height, width = image.size()

    # Calculate differences in the vertical and horizontal directions
    diff_vertical = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
    diff_horizontal = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])

    # Calculate the sum of the absolute differences
    tv_loss = torch.sum(diff_vertical) + torch.sum(diff_horizontal)

    return tv_loss / (batch_size * channels * height * width)


def parameter_analysis(model):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    # 遍历model.parameters()返回的全局参数列表
    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量

    print(f'Total params: {Total_params}', flush=True)
    print(f'Trainable params: {Trainable_params}', flush=True)
    print(f'Non-trainable params: {NonTrainable_params}', flush=True)
