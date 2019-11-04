from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from gridsearch import evaluate_esn_1d, evaluate_esn_2d
from metric import *


def get_time():
    return datetime.now().strftime("%m-%d-%Y %H:%M:%S")


def set_font_sizes():
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('axes', labelsize=16)


def grid_search_input_sparsity(dataset):
    # NB: The keys will always be sorted for reproducibility, so keep them
    # sorted here.
    hidden_nodes = [50, 100, 200]
    sparsity = np.arange(0.1, 1.1, 0.1)
    params = {
        'hidden_nodes': hidden_nodes,
        'w_in_sparsity': sparsity,
    }

    nrmses = evaluate_esn_2d(dataset, params,
                             evaluate_esn_input_sparsity,
                             runs_per_iteration=10)

    labels = ['50 nodes', '100 nodes', '200 nodes']
    set_font_sizes()

    linestyles = ['dotted', 'dashed', 'solid']
    for i, _nrmses in enumerate(nrmses):
        plt.plot(sparsity, np.squeeze(_nrmses), color='black',
                 marker='.', linestyle=linestyles[i], label=labels[i])

    maxlim = np.max(nrmses) + 0.05
    minlim = np.min(nrmses) - 0.05
    plt.ylim(minlim, maxlim)

    plt.ylabel('NRMSE')
    plt.xlabel('Input sparsity')
    plt.legend(fancybox=False, loc='upper left', bbox_to_anchor=(0.0, 1.0))
    plt.hlines(y = np.arange(0.0, 1.05, 0.05), xmin=0.0, xmax=1.0,
               linewidth=0.2)

    maxlim = np.max(nrmses) + 0.15
    minlim = np.min(nrmses) - 0.05
    plt.ylim(minlim, maxlim)

    plt.margins(0.0)
    plt.savefig('plots/' + get_time())
    plt.show()


def grid_search_output_sparsity(dataset):
    # NB: The keys will always be sorted for reproducibility, so keep them
    # sorted here.
    output_sparsity = np.arange(0.0, 1.1, 0.1)
    params = {
        'hidden_nodes': [200],
        'w_out_sparsity': output_sparsity,
    }

    nrmses = evaluate_esn_2d(dataset, params,
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

    nrmses = evaluate_esn_2d(dataset, params,
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
    max_nodes = 200
    min_nodes = 30
    mid_nodes = (max_nodes + min_nodes) // 2

    hidden_nodes = np.arange(min_nodes, max_nodes, 10)
    output_sparsity = np.arange(0.0, 1.05, 0.05)
    params = {
        'hidden_nodes': hidden_nodes,
        'w_out_sparsity': output_sparsity
    }

    nrmses = evaluate_esn_2d(dataset, params,
                             evaluate_esn_output_sparsity,
                             runs_per_iteration=10)

    set_font_sizes()

    sns.heatmap(list(reversed(nrmses)), vmin=0.0, vmax=1.0, square=True)
    ax = plt.axes()

    # Fix half cells at the top and bottom. This is a current bug in Matplotlib.
    ax.set_ylim(ax.get_ylim()[0]+0.5, 0.0)

    x_width = ax.get_xlim()[1]
    y_width = ax.get_ylim()[0]

    plt.xticks([0.0, 0.5*x_width, x_width], [0.0, 0.5, 1.0])
    plt.yticks([0.0, 0.5*y_width, y_width], [max_nodes, mid_nodes, min_nodes])

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    plt.xlabel('Output sparsity')
    plt.ylabel('Reservoir size')
    ax.collections[0].colorbar.set_label('NRMSE')

    plt.show()


def input_noise(dataset):
    pass


def performance_sweep(dataset):
    hidden_nodes = [50, 100, 150, 200]
    params = { 'hidden_nodes': hidden_nodes }
    nrmses = evaluate_esn_1d(dataset, params, runs_per_iteration=10)

    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('axes', labelsize=16)

    plt.plot(hidden_nodes, nrmses, color='black', linestyle='dashed', marker='.')

    plt.ylabel('NARMA10 - NRMSE')
    plt.xlabel('Reservoir size')
    plt.xticks(np.arange(min(hidden_nodes), max(hidden_nodes) + 1, 50))

    maxlim = np.max(nrmses) + 0.05
    minlim = np.min(nrmses) - 0.05
    plt.ylim(minlim, maxlim)

    plt.show()


def visualize(dataset, washout=200):
    u_train, y_train, u_test, y_test = dataset

    esn = ESN(hidden_nodes = 200)
    esn(u_train, y_train)
    y_predicted = esn(u_test)

    target = y_test[washout:]
    predicted = y_predicted

    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('axes', labelsize=16)

    plt.plot(target, 'black', label='Target output')
    plt.plot(predicted, 'red', label='Predicted output', alpha=0.5)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0., fancybox=False)

    plt.ylabel('Reservoir output')
    plt.xlabel('Time')

    plt.show()
