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

import matplotlib.pyplot as plt
from scipy import stats
import tqdm

train_sample_length = 3000
test_sample_length = 2000
n_train_samples = 1
n_test_samples = 1

use_cuda = False
use_cuda = torch.cuda.is_available() if use_cuda else False

# Manual seed for now.
# np.random.seed(2)
# torch.manual_seed(1)

narma10_train_dataset = NARMADataset(train_sample_length, n_train_samples, system_order=10)
narma10_test_dataset = NARMADataset(test_sample_length, n_test_samples, system_order=10)

trainloader = DataLoader(narma10_train_dataset, shuffle=False, num_workers=2)
testloader = DataLoader(narma10_test_dataset, shuffle=False, num_workers=2)

dataiter = iter(trainloader)
train_u, train_y = dataiter.next()
train_u, train_y = Variable(train_u), Variable(train_y)
if use_cuda: train_u, train_y = train_u.cuda(), train_y.cuda()

dataiter = iter(testloader)
test_u, test_y = dataiter.next()
test_u, test_y = Variable(test_u), Variable(test_y)
if use_cuda: test_u, test_y = test_u.cuda(), test_y.cuda()

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


def evaluate_esn(esn, u, y, plot=False):
    y_predicted = esn(u)

    if plot:
        plt.plot(y[0, :, 0], 'black', linestyle='dashed')
        plt.plot(y_predicted[0, :, 0], 'green')
        plt.show()

    mse = echotorch.utils.mse(y_predicted, y)
    nrmse = echotorch.utils.nrmse(y_predicted, y)
    nmse = echotorch.utils.nmse(y_predicted, y)
    return mse, nrmse


def explore_input_connectivity():
    input_connectivity = 100
    reservoir_size = 200
    reservoirs_per_iteration = 10

    mse_list = []
    nrmse_list = []
    connectivities = []

    i = 0
    while input_connectivity <= reservoir_size:
        mse_list.append([])
        nrmse_list.append([])
        connectivities.append(input_connectivity)

        for j in range(reservoirs_per_iteration):
            esn = etnn.ESN(
                input_dim=1,
                hidden_dim=reservoir_size,
                output_dim=1,
                spectral_radius=0.9,
                learning_algo='inv',
                input_connectivity=input_connectivity
            )

            for data in trainloader:
                inputs, targets = data
                inputs, targets = Variable(inputs), Variable(targets)
                if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
                esn(inputs, targets)

            esn.finalize()

            mse, nrmse = evaluate_esn(esn, test_u, test_y.data)
            mse_list[i].append(mse)
            nrmse_list[i].append(nrmse)

        mean_mse = np.mean(mse_list[i])
        mean_nrmse = np.mean(nrmse_list[i])
        print("IC: {}\tMSE: {:.8f}\t NRMSE: {:.8f}".format(input_connectivity, mean_mse, mean_nrmse))

        input_connectivity += 2
        i += 1

    plt.plot(connectivities, np.mean(nrmse_list, axis=1))
    plt.show()


def explore_output_connectivity():
    output_connectivity = 1
    reservoir_size = 50
    reservoirs_per_iteration = 10

    mse_list = [[] for _ in range(reservoir_size)]
    nrmse_list = [[] for _ in range(reservoir_size)]

    for i in tqdm.tqdm(range(reservoir_size)):
        for j in range(reservoirs_per_iteration):
            esn = etnn.ESN(
                input_dim=1,
                hidden_dim=reservoir_size,
                output_dim=1,
                spectral_radius=1.1,
                learning_algo='inv',
                output_connectivity=output_connectivity
            )

            for data in trainloader:
                inputs, targets = data
                inputs, targets = Variable(inputs), Variable(targets)
                if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
                esn(inputs, targets)

            esn.finalize()

            mse, nrmse = evaluate_esn(esn, test_u, test_y.data)
            mse_list[i].append(mse)
            nrmse_list[i].append(nrmse)

        output_connectivity += 1

    plt.plot(np.mean(nrmse_list, axis=1))
    plt.xlabel('Output connectivity')
    plt.ylabel('NRMSE')
    plt.show()


def tune_esn():
    reservoir_size = 200
    input_connectivity = None
    output_connectivity = None

    train_sample_length = 3000
    test_sample_length = 2000

    use_cuda = False
    use_cuda = torch.cuda.is_available() if use_cuda else False

    narma10_train_dataset = NARMADataset(train_sample_length, n_samples=1, system_order=10)
    narma10_test_dataset = NARMADataset(test_sample_length, n_samples=1, system_order=10)

    trainloader = DataLoader(narma10_train_dataset, shuffle=False, num_workers=2)
    testloader = DataLoader(narma10_test_dataset, shuffle=False, num_workers=2)

    dataiter = iter(trainloader)
    train_u, train_y = dataiter.next()
    train_u, train_y = Variable(train_u), Variable(train_y)
    if use_cuda: train_u, train_y = train_u.cuda(), train_y.cuda()

    dataiter = iter(testloader)
    test_u, test_y = dataiter.next()
    test_u, test_y = Variable(test_u), Variable(test_y)
    if use_cuda: test_u, test_y = test_u.cuda(), test_y.cuda()

    esn = etnn.ESN(
        input_dim=1,
        hidden_dim=reservoir_size,
        output_dim=1,
        spectral_radius=0.9,
        learning_algo='inv',
        input_connectivity=input_connectivity,
        output_connectivity=output_connectivity
    )

    for data in trainloader:
        inputs, targets = data
        ninputs, targets = Variable(inputs), Variable(targets)
        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
        esn(inputs, targets)

    esn.finalize()

    mse, nrmse = evaluate_esn(esn, test_u, test_y.data, plot=True)
    print('MSE:', mse)
    print('NRMSE:', nrmse)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='tms RC')
    parser.add_argument('--ic', help='Explore input connectivity', action='store_true')
    parser.add_argument('--oc', help='Explore output connectivity', action='store_true')
    parser.add_argument('--tune', help='Explore tuning of single net', action='store_true')
    args = parser.parse_args()

    if args.ic:
        explore_input_connectivity()
    elif args.oc:
        explore_output_connectivity()
    elif args.tune:
        tune_esn()


if __name__ == '__main__':
    main()
