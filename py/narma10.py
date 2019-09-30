# -*- coding: utf-8 -*-
#
# File : examples/timeserie_prediction/switch_attractor_esn
# Description : NARMA 30 prediction with ESN.
# Date : 26th of January, 2018
#
# This file is part of EchoTorch.  EchoTorch is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Copyright Nils Schaetti <nils.schaetti@unine.ch>

import torch
from echotorch.datasets.NARMADataset import NARMADataset
import echotorch.nn as etnn
import echotorch.utils
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import numpy as np
import mdp

import matplotlib.pyplot as plt
from scipy import stats

train_sample_length = 5000
test_sample_length = 5000
n_train_samples = 1
n_test_samples = 1
spectral_radius = 1.1
leaky_rate = 1.0
input_dim = 1
n_hidden = 500

use_cuda = False
use_cuda = torch.cuda.is_available() if use_cuda else False

# Manual seed for now.
mdp.numx.random.seed(1)
np.random.seed(2)
torch.manual_seed(1)

narma10_train_dataset = NARMADataset(train_sample_length, n_train_samples, system_order=10)
narma10_test_dataset = NARMADataset(test_sample_length, n_test_samples, system_order=10)

trainloader = DataLoader(narma10_train_dataset, shuffle=False, num_workers=2)
testloader = DataLoader(narma10_test_dataset, shuffle=False, num_workers=2)

esn = etnn.LiESN(
    input_dim=input_dim,
    hidden_dim=n_hidden,
    output_dim=1,
    spectral_radius=spectral_radius,
    learning_algo='inv',
    leaky_rate=leaky_rate
)

if use_cuda:
    esn.cuda()

for data in trainloader:
    inputs, targets = data
    inputs, targets = Variable(inputs), Variable(targets)
    if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
    esn(inputs, targets)

esn.finalize()

dataiter = iter(trainloader)
train_u, train_y = dataiter.next()
train_u, train_y = Variable(train_u), Variable(train_y)
if use_cuda: train_u, train_y = train_u.cuda(), train_y.cuda()
y_predicted = esn(train_u)
print(u"Train MSE: {}".format(echotorch.utils.mse(y_predicted.data, train_y.data)))
print(u"Test NRMSE: {}".format(echotorch.utils.nrmse(y_predicted.data, train_y.data)))
print(u"")

dataiter = iter(testloader)
test_u, test_y = dataiter.next()
test_u, test_y = Variable(test_u), Variable(test_y)
if use_cuda: test_u, test_y = test_u.cuda(), test_y.cuda()
y_predicted = esn(test_u)
print(u"Test MSE: {}".format(echotorch.utils.mse(y_predicted.data, test_y.data)))
print(u"Test NRMSE: {}".format(echotorch.utils.nrmse(y_predicted.data, test_y.data)))
print(u"")

noise_mean = 0.0
noise_stddev = 0.2

# ----------
# Just print the difference given a noise distribution.
# ----------

# noise = Variable(test_u.data.new(test_u.size()).normal_(noise_mean, noise_stddev))

# noise_u = test_u + noise
# noise_predicted = esn(noise_u)

# print(u"Noise MSE: {}".format(echotorch.utils.mse(noise_predicted.data, test_y.data)))
# print(u"Noise NRMSE: {}".format(echotorch.utils.nrmse(noise_predicted.data, test_y.data)))
# print(u"")

# plt.plot(test_y.data[0, 2000:2100, 0], 'black', linestyle='dashed')
# plt.plot(y_predicted.data[0, 2000:2100, 0], 'green')
# plt.plot(noise_predicted.data[0, 2000:2100, 0], 'orange')
# plt.show()

# ----------
# How does the error develop with increasing noise?
# ----------

# u_var = test_u.var()
# noise_stddev = 0.00
# stddevs = []
# nrmses = []
# snrs = []

# for i in range(50):
#     noise = Variable(test_u.data.new(test_u.size()).normal_(noise_mean, noise_stddev))

#     noise_u = test_u + noise
#     noise_predicted = esn(noise_u)

#     print(u"Noise MSE: {}".format(echotorch.utils.mse(noise_predicted.data, test_y.data)))
#     print(u"Noise NRMSE: {}".format(echotorch.utils.nrmse(noise_predicted.data, test_y.data)))
#     print(u"")

#     stddevs.append(noise_stddev)
#     snrs.append(10 * np.log10(u_var / (noise_stddev*noise_stddev)))
#     nrmses.append(echotorch.utils.nrmse(noise_predicted.data, test_y.data))

#     noise_stddev += 0.02

# plt.plot(stddevs, nrmses, 'black')
# plt.show()

# plt.plot(snrs, nrmses, 'black')
# plt.show()

# ----------
# SNR stuff.
# ----------

# Information Processing Capacity of Dynamical Systems uses SNR as
# 10*log10(var(u)/var(v)) where u is input signal and v is noise.

# test_u_mean = test_u.mean()
# test_u_var = test_u.var()
# test_u_avg_power = test_u_mean*test_u_mean + test_u_var

# noise_u_mean = noise_mean
# noise_u_var = noise_stddev*noise_stddev
# noise_u_avg_power = noise_mean*noise_mean + noise_u_var

# print("test_u mean:", test_u_mean)
# print("test_u variance:", test_u_var)
# print("test_u average power:", test_u_avg_power)

# print("noise_u mean:", noise_u_mean)
# print("noise_u variance:", noise_u_var)
# print("noise_u average power:", noise_u_avg_power)
