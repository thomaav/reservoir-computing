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
import matplotlib.ticker as mticker
from scipy import stats
import seaborn as sns

use_cuda = False
use_cuda = torch.cuda.is_available() if use_cuda else False

# Manual seed for now.
# np.random.seed(2)
# torch.manual_seed(1)


def narma(train_length, test_length, order=10):
    narma_train_dataset = NARMADataset(train_length, n_samples=1, system_order=10)
    narma_test_dataset = NARMADataset(test_length, n_samples=1, system_order=10)

    trainloader = DataLoader(narma_train_dataset, shuffle=False, num_workers=2)
    testloader = DataLoader(narma_test_dataset, shuffle=False, num_workers=2)

    dataiter = iter(trainloader)
    train_u, train_y = dataiter.next()
    train_u, train_y = Variable(train_u), Variable(train_y)
    if use_cuda: train_u, train_y = train_u.cuda(), train_y.cuda()

    dataiter = iter(testloader)
    test_u, test_y = dataiter.next()
    test_u, test_y = Variable(test_u), Variable(test_y)
    if use_cuda: test_u, test_y = test_u.cuda(), test_y.cuda()

    return train_u, train_y, test_u, test_y


def evaluate_esn(esn, u, y, plot=False, washout=0):
    y_predicted = esn(u)

    if plot:
        plt.plot(y[0, washout:, 0], 'black', linestyle='dashed')
        plt.plot(y_predicted[0, :, 0], 'green')
        plt.show()

    mse = echotorch.utils.mse(y_predicted[0, :, 0], y[0, washout:, 0])
    nrmse = echotorch.utils.nrmse(y_predicted[0, :, 0], y[0, washout:, 0])
    nmse = echotorch.utils.nmse(y_predicted[0, :, 0], y[0, washout:, 0])
    return mse, nrmse, nmse


def explore_input_noise():
    train_length, test_length = 5000, 3000
    test_u, test_y, train_u, train_y = narma(train_length, test_length, order=10)

    # ESN that is not trained with noise.
    esn = etnn.ESN(
        input_dim=1,
        hidden_dim=200,
        output_dim=1,
        spectral_radius=1.0,
        learning_algo='inv',
        win_distrib='gaussian',
        w_distrib='gaussian',
        input_scaling=1.0,
    )

    inputs, targets = Variable(train_u), Variable(train_y)
    if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
    esn(inputs, targets)
    esn.finalize()

    # ESN that is trained with noise.
    noise_esn = etnn.ESN(
        input_dim=1,
        hidden_dim=200,
        output_dim=1,
        spectral_radius=1.0,
        learning_algo='inv',
        win_distrib='gaussian',
        w_distrib='gaussian',
        input_scaling=1.0,
    )

    train_noise_mean = 0.0
    train_noise_stddev = 0.1
    train_noise = Variable(test_u.data.new(train_u.size()).normal_(train_noise_mean, train_noise_stddev))

    inputs, targets = Variable(train_u), Variable(train_y)
    inputs += train_noise
    if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
    noise_esn(inputs, targets)
    noise_esn.finalize()

    u_var = test_u.var()
    noise_mean = 0.0
    noise_stddev = 0.0
    stddevs = []

    nrmses = []
    noise_nrmses = []
    snrs = []

    for i in range(50):
        noise = Variable(test_u.data.new(test_u.size()).normal_(noise_mean, noise_stddev))

        noise_u = test_u + noise
        noise_predicted = esn(noise_u)
        noise_esn_predicted = noise_esn(noise_u)

        print(u"Noise MSE: {}".format(echotorch.utils.mse(noise_predicted.data, test_y.data)))
        print(u"Noise NRMSE: {}".format(echotorch.utils.nrmse(noise_predicted.data, test_y.data)))
        print(u"")

        stddevs.append(noise_stddev)
        snrs.append(10 * np.log10(u_var / (noise_stddev*noise_stddev)))
        nrmses.append(echotorch.utils.nrmse(noise_predicted.data, test_y.data))
        noise_nrmses.append(echotorch.utils.nrmse(noise_esn_predicted.data, test_y.data))

        noise_stddev += 0.02

    plt.plot(stddevs, nrmses, 'black', label='Trained without noise')
    plt.plot(stddevs, noise_nrmses, 'black', linestyle='dashed', label='Trained with noise')
    plt.ylabel('NARMA10 - NRMSE')
    plt.xlabel('Noise standard deviation')
    plt.legend()
    plt.show()

    plt.plot(snrs, nrmses, 'black', label='Trained without noise')
    plt.plot(snrs, noise_nrmses, 'black', linestyle='dashed', label='Trained with noise')
    plt.ylabel('NARMA10 - NRMSE')
    plt.xlabel('SNR')
    plt.legend()
    plt.show()


def explore_input_connectivity():
    reservoir_size = 200
    reservoirs_per_iteration = 10
    train_length, test_length = 5000, 3000
    test_u, test_y, train_u, train_y = narma(train_length, test_length, order=10)

    input_connectivity = 1
    ic_step_size = 10

    mse_list = []
    nrmse_list = []
    connectivities = []

    i = 0
    while input_connectivity <= reservoir_size:
        mse_list.append([])
        nrmse_list.append([])
        connectivities.append(input_connectivity / reservoir_size)

        for j in range(reservoirs_per_iteration):
            esn = etnn.ESN(
                input_dim=1,
                hidden_dim=reservoir_size,
                output_dim=1,
                spectral_radius=1.0,
                learning_algo='inv',
                win_distrib='gaussian',
                w_distrib='gaussian',
                input_scaling=1.0,
                input_connectivity=input_connectivity
            )

            inputs, targets = Variable(train_u), Variable(train_y)
            if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
            esn(inputs, targets)
            esn.finalize()

            mse, nrmse = evaluate_esn(esn, test_u, test_y.data)
            mse_list[i].append(mse)
            nrmse_list[i].append(nrmse)

        mean_mse = np.mean(mse_list[i])
        mean_nrmse = np.mean(nrmse_list[i])
        print("IC: {}\tMSE: {:.8f}\t NRMSE: {:.8f}".format(input_connectivity, mean_mse, mean_nrmse))

        input_connectivity += ic_step_size
        i += 1

    plt.plot(connectivities, np.mean(nrmse_list, axis=1), color='black', marker='.')
    plt.ylabel('NARMA10 - NRMSE')
    plt.xlabel('Input connectivity')
    plt.ylim(0.0, 1.0)
    plt.show()


def explore_output_connectivity():
    reservoir_size = 200
    reservoirs_per_iteration = 10
    train_length, test_length = 5000, 3000
    test_u, test_y, train_u, train_y = narma(train_length, test_length, order=10)

    output_connectivity = 1
    oc_step_size = 10

    mse_list = []
    nrmse_list = []
    connectivities = []

    i = 0
    while output_connectivity - oc_step_size < reservoir_size:
        if output_connectivity > reservoir_size:
            output_connectivity = reservoir_size

        mse_list.append([])
        nrmse_list.append([])
        connectivities.append(output_connectivity / reservoir_size)

        for j in range(reservoirs_per_iteration):
            esn = etnn.ESN(
                input_dim=1,
                hidden_dim=reservoir_size,
                output_dim=1,
                spectral_radius=1.0,
                learning_algo='inv',
                output_connectivity=output_connectivity
            )

            inputs, targets = Variable(train_u), Variable(train_y)
            if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
            esn(inputs, targets)
            esn.finalize()

            mse, nrmse, nmse = evaluate_esn(esn, test_u, test_y.data)
            mse_list[i].append(mse)
            nrmse_list[i].append(nrmse)

        mean_mse = np.mean(mse_list[i])
        mean_nrmse = np.mean(nrmse_list[i])
        print("OC: {}\tMSE: {:.8f}\t NRMSE: {:.8f}".format(output_connectivity, mean_mse, mean_nrmse))

        output_connectivity += oc_step_size
        i += 1

    plt.plot(connectivities, np.mean(nrmse_list, axis=1), color='black', marker='.')
    plt.ylabel('NARMA10 - NRMSE')
    plt.xlabel('Output connectivity')
    plt.ylim(0.0, 1.0)
    plt.show()


def explore_input_connectivity_scaling():
    reservoir_size = 200
    reservoirs_per_iteration = 10
    train_length, test_length = 5000, 3000
    test_u, test_y, train_u, train_y = narma(train_length, test_length, order=10)

    ic_step_size = 10

    input_scaling = 0.1
    input_scaling_step_size = 0.1

    mse_list = []
    nrmse_list = []
    connectivities = []

    while input_scaling <= 2.01:
        print('Input scaling:', input_scaling)
        input_connectivity = 1
        mse_list.insert(0, [])
        nrmse_list.insert(0, [])
        while input_connectivity <= reservoir_size:
            connectivities.append(input_connectivity / reservoir_size)

            it_mse_list = []
            it_nrmse_list = []
            for j in range(reservoirs_per_iteration):
                esn = etnn.ESN(
                    input_dim=1,
                    hidden_dim=reservoir_size,
                    output_dim=1,
                    spectral_radius=1.0,
                    learning_algo='inv',
                    win_distrib='gaussian',
                    w_distrib='gaussian',
                    input_scaling=input_scaling,
                    input_connectivity=input_connectivity
                )

                inputs, targets = Variable(train_u), Variable(train_y)
                if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
                esn(inputs, targets)
                esn.finalize()

                mse, nrmse, nmse = evaluate_esn(esn, test_u, test_y.data)
                it_mse_list.append(mse)
                it_nrmse_list.append(nrmse)

            mean_mse = np.mean(it_mse_list)
            mean_nrmse = np.mean(it_nrmse_list)
            mse_list[0].append(mean_mse)
            nrmse_list[0].append(mean_nrmse)
            print("IC: {}\tMSE: {:.8f}\t NRMSE: {:.8f}".format(input_connectivity, mean_mse, mean_nrmse))
            input_connectivity += ic_step_size
        input_scaling += input_scaling_step_size

    sns.heatmap(nrmse_list, vmin=0.0, vmax=1.0, square=True)
    ax = plt.axes()

    # Fix half cells at the top and bottom.
    ax.set_ylim(ax.get_ylim()[0]+0.5, 0.0)

    x_width = ax.get_xlim()[1]
    y_width = ax.get_ylim()[0]

    plt.xticks([0.0, 0.25*x_width, 0.5*x_width, 0.75*x_width, x_width], ['', 0.25, 0.5, 0.75, ''])
    plt.yticks([0.0, 0.5*y_width, y_width], [2, 1, ''])

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    plt.xlabel('Input connectivity')
    plt.ylabel('Input scaling')

    plt.show()


def explore_partial_visibility():
    reservoirs_per_iteration = 10
    train_length, test_length = 5000, 3000
    test_u, test_y, train_u, train_y = narma(train_length, test_length, order=10)

    initial_output_connectivity = 30
    output_connectivity = 30
    oc_step_size = 10
    reservoir_size_step_size = 10

    mse_list = []
    nrmse_list = []
    reservoir_sizes = []

    while output_connectivity <= 200:
        print('Output connectivity:', output_connectivity)
        reservoir_size = output_connectivity

        offset = int((output_connectivity-initial_output_connectivity)/oc_step_size)
        mse_list.insert(0, [0]*offset)
        nrmse_list.insert(0, [0]*offset)

        while reservoir_size <= 200:
            reservoir_sizes.append(reservoir_size)

            mses = []
            nrmses = []

            for j in range(reservoirs_per_iteration):
                esn = etnn.ESN(
                    input_dim=1,
                    hidden_dim=reservoir_size,
                    output_dim=1,
                    spectral_radius=1.0,
                    learning_algo='inv',
                    win_distrib='gaussian',
                    w_distrib='gaussian',
                    output_connectivity=output_connectivity
                )

                inputs, targets = Variable(train_u), Variable(train_y)
                if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
                esn(inputs, targets)
                esn.finalize()

                mse, nrmse, nmse = evaluate_esn(esn, test_u, test_y.data)
                mses.append(mse)
                nrmses.append(nrmse)

            mean_mse = np.mean(mses)
            mean_nrmse = np.mean(nrmses)
            mse_list[0].append(mean_mse)
            nrmse_list[0].append(mean_nrmse)
            print("Size: {}\tMSE: {:.8f}\t NRMSE: {:.8f}".format(reservoir_size, mean_mse, mean_nrmse))

            reservoir_size += reservoir_size_step_size
        output_connectivity += oc_step_size

    mask = np.zeros_like(nrmse_list, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    mask = np.fliplr(mask)

    sns.heatmap(nrmse_list, vmin=0.0, vmax=1.0, square=True, mask=mask)
    ax = plt.axes()

    # Fix half cells at the top and bottom.
    ax.set_ylim(ax.get_ylim()[0]+0.5, 0.0)

    x_width = ax.get_xlim()[1]
    y_width = ax.get_ylim()[0]

    plt.xticks([0.0, 0.5*x_width, x_width], [30, 115, 200])
    plt.yticks([0.0, 0.5*y_width, y_width], [200, 115, 30])

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    plt.xlabel('Reservoir size')
    plt.ylabel('Output connectivity')

    plt.show()


def tune_esn():
    washout = 200
    train_sample_length = 2000 + washout
    test_sample_length = 3000 + washout

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
        hidden_dim=200,
        output_dim=1,
        spectral_radius=0.9,
        nonlin_func=torch.tanh,
        learning_algo='inv',
        win_distrib='gaussian',
        w_distrib='gaussian',
        input_scaling=1.0,
        w_sparsity=0.2,
        # input_connectivity=30,
        washout=washout,
        # output_connectivity=None
    )

    for data in trainloader:
        inputs, targets = data
        inputs, targets = Variable(inputs), Variable(targets)
        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
        esn(inputs, targets)

    esn.finalize()

    # mse, nrmse, nmse = evaluate_esn(esn, train_u, train_y.data, plot=True)
    # print('Train MSE:', mse)
    # print('Train NRMSE:', nrmse)
    # print('Train NMSE:', nmse)
    # print()
    mse, nrmse, nmse = evaluate_esn(esn, test_u, test_y.data, plot=True, washout=washout)
    print('Test MSE:', mse)
    print('Test NRMSE:', nrmse)
    print('Test NMSE:', nmse)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='tms RC')
    parser.add_argument('--ic', help='Explore input connectivity', action='store_true')
    parser.add_argument('--oc', help='Explore output connectivity', action='store_true')
    parser.add_argument('--tune', help='Explore tuning of single net', action='store_true')
    parser.add_argument('--scale', help='Scale connectivity', action='store_true')
    parser.add_argument('--partial', help='Explore partial visibility', action='store_true')
    parser.add_argument('--input_noise', help='Explore input noise', action='store_true')
    parser.add_argument('--output_noise', help='Explore output noise', action='store_true')
    args = parser.parse_args()

    if args.ic:
        if args.scale:
            explore_input_connectivity_scaling()
        else:
            explore_input_connectivity()
    elif args.oc:
        explore_output_connectivity()
    elif args.tune:
        tune_esn()
    elif args.partial:
        explore_partial_visibility()
    elif args.input_noise:
        explore_input_noise()
    elif args.output_noise:
        explore_output_noise()


if __name__ == '__main__':
    main()
