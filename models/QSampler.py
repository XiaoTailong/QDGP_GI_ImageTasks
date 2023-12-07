import os
import time
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import tensorcircuit as tc

from models.model import quantum_circuit_fixing, quantum_circuit_Noise, quantum_circuit_Noise_001
# with tensorflow backend and pytorch interface
K = tc.set_backend("tensorflow")

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


class QCBMPrior(nn.Module):
    """
    1. The model is used to sample the discrete bit strings as the generative prior information
    2. Sample the QCBM as the input of Classical generators such as BigGan model or DCGAN, etc
    3. Deep quantum generative prior (DQGP) is quite interesting!
    """
    def __init__(self, nqubits, nlayers, measurement_setting=None):
        super(QCBMPrior, self).__init__()
        self.nqubits = nqubits
        self.nlayers = nlayers
        self.measurement_setting = measurement_setting
        # if measurement_setting == 'o':
        qpred_vmap = K.vmap(quantum_circuit_Noise, vectorized_argnums=0)
        self.params = torch.nn.Parameter(0.1 * torch.randn([nlayers, nqubits, 4]))
        self.q_weights_final = torch.nn.Parameter(0.1 * torch.randn([nqubits, 3]))
        # self.q_weights_basis = torch.nn.Parameter(0.1 * torch.randn([nqubits, ]))

        self.qpred_batch = tc.interfaces.torch_interface(qpred_vmap, jit=True)
        # self.linear = nn.Linear(self.nqubits, 120)

    def forward(self, inputs, temperature=1):
        """
        :param inputs: may be zeros for non-input mode, may be the output of CNN low dimension vector
        Args:
            temperature: temperature is infinity, then uniform distribution. Infinity is zero, then totally equivalent
        """
        # B, _ = inputs.shape
        if self.measurement_setting == "o":  # 不进行学习，只采样
            expec_obs = self.qpred_batch(inputs, self.params, self.q_weights_final, self.q_weights_basis)
            samples = prior_sampling(expec_obs)
            return samples
        elif self.measurement_setting == 't':  # 离散分布的连续化，松弛
            expec_obs = self.qpred_batch(inputs, self.params, self.q_weights_final, self.q_weights_basis)
            # expec_obs = expec_obs[:, :self.nqubits].repeat(1, int(120/self.nqubits))  # 8 * 15
            soft_samples = prior_sampling_soft(expec_obs[:, :self.nqubits], 1.0)
            soft_samples = self.linear(soft_samples)
            # samples = prior_sampling(expec_obs)
            return soft_samples
        elif self.measurement_setting == 'd' or self.measurement_setting == 's':
            expec_obs = self.qpred_batch(inputs, self.params, self.q_weights_final)

            # 直接根据概率进行训练 相当于softmax
            return expec_obs


def prior_sampling(expec_obs: torch.Tensor) -> torch.Tensor:
    r"""
    Simulate the sampling process of the prior distribution, from QCBM.
    Args:
        expec_obs: The vector of expectation values whose length is the same as 'latent_dim'.
        batch_size: The number of samples in a batch.

    Returns:
        The sampling results, values in {-1, 1}.
    """
    batch_size, dims = expec_obs.shape
    prior_samples = torch.rand([batch_size, dims]).cuda()  # 随机节点在这里，但是不参与训练。
    expec_obs_probs = (expec_obs + 1) / 2.
    prior_samples = (expec_obs_probs - prior_samples >= 0).type(torch.cuda.FloatTensor)  # 这个地方不可导。
    # prior_samples = prior_samples * 2 - 1  # [-1与1的数值]
    return prior_samples


def prior_sampling_soft(expec_obs: torch.Tensor, temperature) -> torch.Tensor:
    r"""

    Args:
        expec_obs: the expectation values of Pauli Z operator
        temperature: the coefficient to control the soft relaxation
    Returns:
        samples with relaxation
    """
    batch_size, dims = expec_obs.shape
    prior_samples = torch.rand([batch_size, dims]).cuda()  # 随机节点在这里，但是不参与训练。
    expec_obs_probs = (1 - expec_obs) / 2.  # the probability of measure 1 using Z measurement
    sigmoid_inverse = -1 * torch.log(1/(prior_samples + 1e-8) - 1)
    soft_samples = torch.special.expit((expec_obs_probs + sigmoid_inverse) / temperature)

    return soft_samples

class CNN(nn.Module):
    def __init__(self, latent_dim):
        super(CNN, self).__init__()
        # define CNN to extract features of DGI image
        self.channels = 128
        self.d2 = self.channels // 2
        self.d4 = self.channels // 4
        self.d8 = self.channels // 8

        self.conv = nn.Sequential(
            nn.Conv2d(1, self.d8, kernel_size=3, stride=1, padding='same'),  # [-1, 16, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(self.d8, self.d4, 3, 1, padding='same'),  # [-1, 32, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(self.d8, self.d2, 3, 1, padding='same'),  # [-1, 64, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(self.d2, self.d2, 3, 1, padding='same'),  # [-1, 64, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(self.d2, self.d2, 3, 1, padding='same'),  # [-1, 64, 4, 4]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.d2 * 4 * 4, self.d4 * 3 * 3),  # [-1, 32*9]
            nn.ReLU(),
            nn.Linear(self.d4 * 3 * 3, latent_dim),  # [-1, 16]
            nn.Tanh(),
        )

    def forward(self, inputs):

        outputs = self.conv(inputs)

        output = self.fc(outputs)

        output = (output - torch.max(output)) / (torch.max(output) - torch.min(output))

        return output


class ANN(nn.Module):
    def __init__(self, latent_dim, n_measurements):
        super(ANN, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(n_measurements, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            # nn.Tanh(), # 暂时先使用tanh将输出数据进行归一化，得到输出。当然也可以使用最小最大归一化方式得到输出
        )

    def forward(self, inputs):

        output = self.linear(inputs)
        output = (output - torch.max(output)) / (torch.max(output) - torch.min(output))
        return output


class EnsembleQCBM(nn.Module):
    def __init__(self, N_heads, nqubits, nlayers, measurement_setting=None):
        super(EnsembleQCBM, self).__init__()
        self.Neads = N_heads
        self.nqubits = nqubits
        self.nlayers = nlayers
        self.measurement_setting = measurement_setting
        if measurement_setting == 'd':
            self.QNN_list = nn.ModuleList([QCBMPrior(nqubits, nlayers, measurement_setting)
                                           for _ in range(N_heads)])
        else:
            self.QNN = QCBMPrior(nqubits, nlayers, measurement_setting)

    def forward(self, inputs, temperature):
        # initlize a tensor

        # 每个circuit输出的概率为40维度的概率分布。而不是单纯的z分布。这样对QNN的训练压力会降低。
        if self.measurement_setting == "d":
            output = []
            for i, QNN in enumerate(self.QNN_list):
                # if ideal case, uncomment following two lines
                seeds = torch.rand(size=(1, self.nqubits * 2)).cuda()
                inputs_each = inputs[:, i*self.nqubits:(i+1)*self.nqubits]
                inputs_each = torch.cat([inputs_each, seeds], dim=1)

                output.append(QNN(inputs_each, temperature))

            latent_samples = torch.cat(output, dim=1)
        # print("the shape of the quantum sample", latent_samples.shape)
            return latent_samples
        else:
            latent_samples = self.QNN(inputs, temperature)  # inputs不同啦
            return latent_samples


class EnsembleQCBM_single(nn.Module):
    def __init__(self, N_heads, nqubits, nlayers, measurement_setting=None):
        super(EnsembleQCBM_single, self).__init__()
        self.Neads = N_heads
        self.nqubits = nqubits
        self.nlayers = nlayers
        self.measurement_setting = measurement_setting
        self.QNN = QCBMPrior(nqubits, nlayers, measurement_setting)

    def forward(self, inputs, temperature):
        # initlize a tensor
        # 每个circuit输出的概率为40维度的概率分布。而不是单纯的z分布。这样对QNN的训练压力会降低。
        if self.measurement_setting == "s":
            inputs = torch.reshape(inputs, [6, self.nqubits])
            # if noisy otherwise comment following two lines
            seeds = torch.rand(size=(1, self.nqubits * 2)).repeat(6, 1).cuda()
            inputs = torch.cat([inputs, seeds], dim=1)

            latent_samples = self.QNN(inputs, temperature)
            latent_samples = torch.reshape(latent_samples, [1, 120])
        # print("the shape of the quantum sample", latent_samples.shape)
            return latent_samples
        else:
            latent_samples = self.QNN(inputs, temperature)  # inputs不同啦
            return latent_samples


class EnsembleQCBM_0001(nn.Module):
    def __init__(self, N_heads, nqubits, nlayers, measurement_setting=None):
        super(EnsembleQCBM_0001, self).__init__()
        self.Neads = N_heads
        self.nqubits = nqubits
        self.nlayers = nlayers
        self.measurement_setting = measurement_setting
        if measurement_setting == 'd':
            self.QNN_list = nn.ModuleList([QCBMPrior001(nqubits, nlayers, measurement_setting)
                                           for _ in range(N_heads)])
        else:
            self.QNN = QCBMPrior001(nqubits, nlayers, measurement_setting)

    def forward(self, inputs, temperature):
        # initlize a tensor

        # 每个circuit输出的概率为40维度的概率分布。而不是单纯的z分布。这样对QNN的训练压力会降低。
        if self.measurement_setting == "d":
            output = []
            for i, QNN in enumerate(self.QNN_list):
                # if ideal case, uncomment following two lines
                seeds = torch.rand(size=(1, self.nqubits * 2)).cuda()
                inputs_each = inputs[:, i*self.nqubits:(i+1)*self.nqubits]
                inputs_each = torch.cat([inputs_each, seeds], dim=1)

                output.append(QNN(inputs_each, temperature))

            latent_samples = torch.cat(output, dim=1)
        # print("the shape of the quantum sample", latent_samples.shape)
            return latent_samples
        else:
            latent_samples = self.QNN(inputs, temperature)  # inputs不同啦
            return latent_samples


class EnsembleQCBM_single_0001(nn.Module):
    def __init__(self, N_heads, nqubits, nlayers, measurement_setting=None):
        super(EnsembleQCBM_single_0001, self).__init__()
        self.Neads = N_heads
        self.nqubits = nqubits
        self.nlayers = nlayers
        self.measurement_setting = measurement_setting
        self.QNN = QCBMPrior001(nqubits, nlayers, measurement_setting)

    def forward(self, inputs, temperature):
        # initlize a tensor
        # 每个circuit输出的概率为40维度的概率分布。而不是单纯的z分布。这样对QNN的训练压力会降低。
        if self.measurement_setting == "s":
            inputs = torch.reshape(inputs, [6, self.nqubits])
            # if noisy otherwise comment following two lines
            seeds = torch.rand(size=(1, self.nqubits * 2)).repeat(6, 1).cuda()
            inputs = torch.cat([inputs, seeds], dim=1)

            latent_samples = self.QNN(inputs, temperature)
            latent_samples = torch.reshape(latent_samples, [1, 120])
        # print("the shape of the quantum sample", latent_samples.shape)
            return latent_samples
        else:
            latent_samples = self.QNN(inputs, temperature)  # inputs不同啦
            return latent_samples


class QCBMPrior001(nn.Module):
    """
    1. The model is used to sample the discrete bit strings as the generative prior information
    2. Sample the QCBM as the input of Classical generators such as BigGan model or DCGAN, etc
    3. Deep quantum generative prior (DQGP) is quite interesting!
    """
    def __init__(self, nqubits, nlayers, measurement_setting=None):
        super(QCBMPrior001, self).__init__()
        self.nqubits = nqubits
        self.nlayers = nlayers
        self.measurement_setting = measurement_setting
        # if measurement_setting == 'o':
        qpred_vmap = K.vmap(quantum_circuit_Noise_001, vectorized_argnums=0)
        self.params = torch.nn.Parameter(0.1 * torch.randn([nlayers, nqubits, 4]))
        self.q_weights_final = torch.nn.Parameter(0.1 * torch.randn([nqubits, 3]))
        # self.q_weights_basis = torch.nn.Parameter(0.1 * torch.randn([nqubits, ]))

        self.qpred_batch = tc.interfaces.torch_interface(qpred_vmap, jit=True)
        # self.linear = nn.Linear(self.nqubits, 120)

    def forward(self, inputs, temperature=1):
        """
        :param inputs: may be zeros for non-input mode, may be the output of CNN low dimension vector
        Args:
            temperature: temperature is infinity, then uniform distribution. Infinity is zero, then totally equivalent
        """
        # B, _ = inputs.shape
        if self.measurement_setting == "o":  # 不进行学习，只采样
            expec_obs = self.qpred_batch(inputs, self.params, self.q_weights_final, self.q_weights_basis)
            samples = prior_sampling(expec_obs)
            return samples
        elif self.measurement_setting == 't':  # 离散分布的连续化，松弛
            expec_obs = self.qpred_batch(inputs, self.params, self.q_weights_final, self.q_weights_basis)
            # expec_obs = expec_obs[:, :self.nqubits].repeat(1, int(120/self.nqubits))  # 8 * 15
            soft_samples = prior_sampling_soft(expec_obs[:, :self.nqubits], 1.0)
            soft_samples = self.linear(soft_samples)
            # samples = prior_sampling(expec_obs)
            return soft_samples
        elif self.measurement_setting == 'd' or self.measurement_setting == 's':
            expec_obs = self.qpred_batch(inputs, self.params, self.q_weights_final)

            # 直接根据概率进行训练 相当于softmax
            return expec_obs