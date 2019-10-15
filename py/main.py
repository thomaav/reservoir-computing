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
    sparsity = np.arange(0.0, 1.1, 0.1)
    params = {
        'hidden_nodes': [200],
        'w_out_sparsity': sparsity,
    }

    nrmses = evaluate_esn_grid_search_2d(dataset, params,
                                         evaluate_esn_output_sparsity,
                                         runs_per_iteration=10)

    plt.plot(sparsity, np.squeeze(nrmses), color='black', marker='.')
    plt.ylabel('NARMA10 - NRMSE')
    plt.xlabel('Output sparsity')
    plt.ylim(0.0, 1.0)
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

    plt.xlabel('Output connectivity')
    plt.ylabel('Reservoir size')
    ax.collections[0].colorbar.set_label('NRMSE')

    plt.show()


def run_single_esn(dataset):
    esn = ESN(
        hidden_nodes=200,
        w_out_sparsity=0.7
    )

    print('NRMSE:', evaluate_esn(dataset, esn))


def main():
    u_train, y_train = NARMA(sample_len = 2000)
    u_test, y_test = NARMA(sample_len = 3000)
    dataset = [u_train, y_train, u_test, y_test]

    import argparse
    parser = argparse.ArgumentParser(description='tms RC')
    parser.add_argument('--input_sparsity', help='Explore input sparsity', action='store_true')
    parser.add_argument('--output_sparsity', help='Explore output sparsity', action='store_true')
    parser.add_argument('--partial', help='Explore partial visibility', action='store_true')
    parser.add_argument('--single_esn', help='Test performance of a singles ESN', action='store_true')
    args = parser.parse_args()

    if args.input_sparsity:
        grid_search_input_sparsity(dataset)
    elif args.output_sparsity:
        grid_search_output_sparsity(dataset)
    elif args.partial:
        grid_search_partial_visibility(dataset)
    elif args.single_esn:
        run_single_esn(dataset)


if __name__ == '__main__':
    main()
