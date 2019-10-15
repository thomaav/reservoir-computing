import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import ParameterGrid
import seaborn as sns
import torch

from metric import *
from dataset import NARMA


def evaluate_esn_grid_search_2d(dataset, params, eval_func, runs_per_iteration=1):
    if len(params.keys()) != 2:
        class InvalidParameterDictException(Exception):
            pass
        raise InvalidParameterDictException('2d grid search requires exactly 2 parameter lists')

    sorted_params = sorted(params.keys())
    p1 = sorted_params[0]
    p2 = sorted_params[1]
    n_p1, n_p2 = len(params[p1]), len(params[p2])
    output = [[] for _ in range(n_p1)]

    param_grid = ParameterGrid(params)
    for i, params in enumerate(param_grid):
        print(params)
        p1_i = i // n_p2

        nrmses = []
        for i in range(runs_per_iteration):
            nrmse = eval_func(dataset, params[p1], params[p2])
            nrmses.append(nrmse)
        output[p1_i].append(np.mean(nrmses))

    return output


def grid_search_input_sparsity(dataset):
    # NB: The keys will always be sorted for reproducibility, so keep them
    # sorted here.
    sparsity = np.arange(0.0, 1.1, 0.1)
    params = {
        'hidden_nodes': [200],
        'w_in_sparsity': sparsity,
    }

    nrmses = evaluate_esn_grid_search_2d(dataset, params,
                                         evaluate_esn_input_sparsity,
                                         runs_per_iteration=10)

    plt.plot(sparsity, np.squeeze(nrmses), color='black', marker='.')
    plt.ylabel('NARMA10 - NRMSE')
    plt.xlabel('Input sparsity')
    plt.ylim(0.0, 1.0)
    plt.show()


def grid_search_output_sparsity(dataset):
    # NB: The keys will always be sorted for reproducibility, so keep them
    # sorted here.
    output_sparsity = np.arange(0.0, 1.1, 0.1)
    params = {
        'hidden_nodes': [200],
        'w_out_sparsity': output_sparsity,
    }

    nrmses = evaluate_esn_grid_search_2d(dataset, params,
                                         evaluate_esn_output_sparsity,
                                         runs_per_iteration=10)

    plt.plot(sparsity, np.squeeze(nrmses), color='black', marker='.')
    plt.ylabel('NARMA10 - NRMSE')
    plt.xlabel('Output sparsity')
    plt.ylim(0.0, 1.0)
    plt.show()


def grid_search_input_sparsity_input_scaling(dataset):
    input_scaling = np.arange(0.0, 2.1, 0.1)
    input_sparsity = np.arange(0.0, 1.05, 0.05)
    params = {
        'input_scaling': input_scaling,
        'w_in_sparsity': input_sparsity,
    }

    nrmses = evaluate_esn_grid_search_2d(dataset, params,
                                         evaluate_esn_input_sparsity_scaling,
                                         runs_per_iteration=10)

    sns.heatmap(list(reversed(nrmses)), vmin=0.0, vmax=1.0, square=True)
    ax = plt.axes()

    # Fix half cells at the top and bottom. This is a current bug in Matplotlib.
    ax.set_ylim(ax.get_ylim()[0]+0.5, 0.0)

    x_width = ax.get_xlim()[1]
    y_width = ax.get_ylim()[0]

    plt.xticks([0.0, 0.5*x_width, x_width], [0.0, 0.5, 1.0])
    plt.yticks([0.0, 0.5*y_width, y_width], [2, 1, ''])

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    plt.xlabel('Input sparsity')
    plt.ylabel('Input scaling')
    ax.collections[0].colorbar.set_label('NRMSE')

    plt.show()


def grid_search_partial_visibility(dataset):
    hidden_nodes = np.arange(30, 200, 10)
    output_sparsity = np.arange(0.0, 1.1, 0.1)
    params = {
        'hidden_nodes': hidden_nodes,
        'w_out_sparsity': output_sparsity
    }

    nrmses = evaluate_esn_grid_search_2d(dataset, params,
                                         evaluate_esn_output_sparsity,
                                         runs_per_iteration=10)

    sns.heatmap(list(reversed(nrmses)), vmin=0.0, vmax=1.0, square=True)
    ax = plt.axes()

    # Fix half cells at the top and bottom. This is a current bug in Matplotlib.
    ax.set_ylim(ax.get_ylim()[0]+0.5, 0.0)

    x_width = ax.get_xlim()[1]
    y_width = ax.get_ylim()[0]

    plt.xticks([0.0, 0.5*x_width, x_width], [0.0, 0.5, 1.0])
    plt.yticks([0.0, 0.5*y_width, y_width], [200, 115, 30])

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    plt.xlabel('Output sparsity')
    plt.ylabel('Reservoir size')
    ax.collections[0].colorbar.set_label('NRMSE')

    plt.show()


def input_noise(dataset):
    pass


def run_single_esn(dataset):
    esn = ESN(
        hidden_nodes=200,
        input_scaling=2.0,
    )

    print('NRMSE:', evaluate_esn(dataset, esn))


def main():
    u_train, y_train = NARMA(sample_len = 2000)
    u_test, y_test = NARMA(sample_len = 3000)
    dataset = [u_train, y_train, u_test, y_test]

    import argparse
    parser = argparse.ArgumentParser(description='tms RC')
    parser.add_argument('--input_sparsity', action='store_true')
    parser.add_argument('--input_sparsity_scaling', action='store_true')
    parser.add_argument('--output_sparsity', action='store_true')
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--input_noise', action='store_true')
    parser.add_argument('--single_esn', action='store_true')
    args = parser.parse_args()

    if args.input_sparsity:
        grid_search_input_sparsity(dataset)
    if args.input_sparsity_scaling:
        grid_search_input_sparsity_input_scaling(dataset)
    elif args.output_sparsity:
        grid_search_output_sparsity(dataset)
    elif args.partial:
        grid_search_partial_visibility(dataset)
    elif args.input_noise:
        input_noise(dataset)
    elif args.single_esn:
        run_single_esn(dataset)


if __name__ == '__main__':
    main()
