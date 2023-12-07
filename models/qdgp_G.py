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

# from models.QSampler import EnsembleQCBM
import models
import utils
from models.downsampler import Downsampler


class QDGP_G(object):
    def __init__(self, config):
        # self.target_bucket = None
        self.rank, self.world_size = 0, 1
        if config['dist']:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        self.config = config

        # 这里记得改一下
        self.factor = 1 if config['resolution'] == 128 else 2  # 128 * 128 image resolution
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
            self.G = models.Generator(**config).cuda()
        else:
            self.G = models.Generator(**config).cuda()
        self.D = None  # 不需要NN判别器，有物理损失
        # self.G.optim = torch.optim.Adam(
        #     [{'params': self.G.get_params(i, self.update_embed)}
        #         for i in range(len(self.G.blocks) + 1)],
        #     lr=config['G_lrs'][0],  # 采用learning rate 固定的方式 不进行learning rate自适应调节
        #     )

        self._prepare_latent()
        # load weights, we can choose different mode to compare the pretraing and non-pretraining
        if config['random_G']:
            self.random_G()
        else:
            utils.load_weights(
                self.G if not (config['use_ema']) else None,
                self.D,  # D is none and has no effect
                config['weights_root'],
                name_suffix=config['load_weights'],
                G_ema=self.G if config['use_ema'] else None,
                strict=False)

        self.G.eval()
        if self.D is not None:  # D is none
            self.D.eval()
        self.G_weight = deepcopy(self.G.state_dict())

        # Downsampler for producing low-resolution image
        self.downsampler = Downsampler(
            n_planes=3,
            factor=self.factor,
            kernel_type='lanczos2',
            phase=0.5,
            preserve_size=True).type(torch.cuda.FloatTensor)

        # prepare latent variable and optimizer
        # self._prepare_quantum_latent()
        # prepare learning rate scheduler
        # self.G_scheduler = utils.LRScheduler(self.G.optim, config['warm_up'])
        # self.z_scheduler = utils.LRScheduler(self.z_optim, config['warm_up'])

        self.criterion = utils.PhysicalLoss()

    def _prepare_latent(self):
        self.z = torch.zeros((1, self.G.dim_z)).normal_().cuda()
        self.z = Variable(self.z, requires_grad=True)

        self.y = torch.zeros(1).long().cuda()


    def reset_G(self):
        self.G.load_state_dict(self.G_weight, strict=False)
        self.G.reset_in_init()
        if self.config['random_G']:
            self.G.train()
        else:
            self.G.eval()

    def random_G(self):
        self.G.init_weights()

    def set_target(self, target, category):
        self.target_bucket = target
        self.y.fill_(category.item())

    def run(self, patterns, bucket_target, save_interval=None):

        loss_dict = {}
        curr_step = 0

        # parameter_analysis(self.G)
        loss_list = []
        for stage, iteration in enumerate(self.iterations):
            # 分了不同的阶段，每个阶段都进行无训练学习
            self.G.optim = torch.optim.Adam(
                [{'params': self.G.get_params(i, self.update_embed)}
                 for i in range(len(self.G.blocks) + 1)],
                lr=self.config['G_lrs'][stage]  # 采用learning rate 固定的方式 不进行learning rate自适应调节
            )

            self.z_optim = torch.optim.Adam(
                [{'params': self.z, 'lr': self.config['z_lrs'][stage]}],
                betas=(self.config['G_B1'], self.config['G_B2']),
                weight_decay=0,
                eps=1e-8
            )

            for i in range(iteration):
                start = time.time()
                curr_step += 1
                # setup learning rate
                # self.G_scheduler.update(curr_step, self.G_lrs[stage])
                # self.z_scheduler.update(curr_step, self.z_lrs[stage])
                self.z_optim.zero_grad()

                if self.update_G:
                    self.G.optim.zero_grad()

                rec_image = self.G(self.z, self.G.shared(self.y), use_in=self.use_in[stage])
                save_z = self.z.clone()
                rec_image = self.downsampler((rec_image + 1) / 2)  # downsample to (1, 1, 64, 64)
                # rec_image = transforms.Grayscale()(rec_image*2-1)  # with shape (1, 1, 64, 64)
                # rec_image = (rec_image + 1) / 2
                _, _, width, height = rec_image.shape

                tv_loss = 1e-5 * total_variation_loss(image=rec_image)

                bucket = PhysicalForward(rec_image, patterns)

                # print(bucket.shape, flush=True)
                # print(self.target_bucket.shape, flush=True)
                bucket_loss = F.mse_loss(bucket, bucket_target.cuda())/(width**2)
                # 加入了TV正则化
                loss = bucket_loss + tv_loss
                loss.backward()

                self.z_optim.step()

                if self.update_G:
                    self.G.optim.step()
                loss_list.append(loss.detach().cpu().numpy())
                loss_dict = {'mse_loss': bucket_loss}
                end = time.time()
                if i == 0 or (i + 1) % self.config['print_interval'] == 0:
                    if self.rank == 0:
                        print(', '.join(
                            ['Stage: [{0}/{1}]'.format(stage + 1, len(self.iterations))] +
                            ['Iter: [{0}/{1}]'.format(i + 1, iteration)] +
                            ['%s : %+4.8f' % (key, loss_dict[key]) for key in loss_dict] +
                            ['Cost: {:+.8f}'.format(end-start)]
                        ), flush=True)

                        if self.config['random_G']:
                            np.save("./experiments/data_recon/{}/DGP128_recons_dim{}_iter{}.npy"
                                    .format(self.object, self.dim, curr_step),
                                    np.array(loss_list))
                            self.to_img(rec_image.squeeze().detach().cpu().numpy(), curr_step, True)

                            np.save("./experiments/data_recon/{}/DGP128_prior_dim{}_iter{}.npy"
                                    .format(self.object, self.dim, curr_step),
                                    save_z.detach().cpu().numpy())

                        else:
                            np.save("./experiments/data_recon/{}/DGP128_pretrain_recons_dim{}_iter{}.npy"
                                    .format(self.object, self.dim,
                                            curr_step, False),
                                    np.array(loss_list))
                            self.to_img(rec_image.squeeze().detach().cpu().numpy(), curr_step, False)
                            np.save("./experiments/data_recon/{}/DGP128_pretrain_prior_dim{}_iter{}.npy"
                                    .format(self.object, self.dim,
                                            curr_step),
                                    save_z.detach().cpu().numpy())


                # # stop the reconstruction if the loss reaches a threshold
                # if mse_loss.item() < self.config['stop_mse']:
                #     break

        return loss_dict

    # TODO
    def select_z(self, patterns, select_y=False):
        with torch.no_grad():
            # untrain mode, y is random
            z_all, y_all, loss_all = [], [], []
            if self.rank == 0:
                print('Selecting z from {} samples'.format(self.select_num), flush=True)

            for i in range(self.select_num):
                # model initilization has self.z
                # here 切换为整数，而不是浮点数先验信息
                self.z.normal_(mean=0, std=self.config['sample_std'])
                z_all.append(self.z.cpu())
                if select_y:
                    self.y.random_(0, self.config['n_classes'])
                    y_all.append(self.y.cpu())
                recon_images = self.G(self.z, self.G.shared(self.y))
                # 这里需要将图像重新归一化，因为输出是【-1，1】
                recon_images = self.downsampler((recon_images + 1) / 2)
                # recon_images = transforms.Grayscale()(recon_images)  # transform into gray image
                bucket = PhysicalForward(recon_images, patterns)
                # print("the bucket shape is:", bucket.shape, flush=True)
                # print("the target shape is:", self.target_bucket.shape, flush=True)
                bucket_loss = self.criterion(bucket, self.target_bucket.cuda())

                loss = bucket_loss + 1e-5*total_variation_loss(recon_images)
                loss_all.append(loss.view(1).cpu())
                if self.rank == 0 and (i + 1) % 100 == 0:
                    print('Generating {}th sample'.format(i + 1), flush=True)
            loss_all = torch.cat(loss_all)
            idx = torch.argmin(loss_all)
            # choose the minimum loss of z
            # our quantum net does not support the copy operation ?
            # as long as it is tensors it supports.
            self.z.copy_(z_all[idx])

            if select_y:
                self.y.copy_(y_all[idx])

    def to_img(self, images, iterations, random_G):
        # images = np.squeeze(images)
        images = (images - np.min(images)) / (np.max(images) - np.min(images))
        images = images * 255.0
        im = Image.fromarray(images)
        im = im.convert("L")
        if random_G:
            im.save("./experiments/data_recon/{}/DGP128_recons_dim{}_iter{}.jpeg"
                    .format(self.object,
                            self.dim,
                            iterations))
        else:
            im.save("./experiments/data_recon/{}/DGP128_pretrain_recons_dim{}_iter{}.jpeg"
                    .format(self.object,
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

    return tv_loss/(batch_size*channels*height*width)


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