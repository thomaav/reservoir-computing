from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

from ESN import Distribution
from gridsearch import evaluate_esn_1d, evaluate_esn_2d
from metric import *


def get_time():
    return datetime.now().strftime("%m-%d-%Y %H:%M:%S")


def set_font_sizes():
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('axes', labelsize=16)


def grid_search_input_density(dataset):
    # NB: The keys will always be sorted for reproducibility, so keep them
    # sorted here.
    hidden_nodes = [50, 100, 200]
    density = np.arange(0.1, 1.1, 0.1)
    params = {
        'hidden_nodes': hidden_nodes,
        'w_in_density': density,
    }

    nrmses = evaluate_esn_2d(dataset, params,
                             evaluate_esn_input_density,
                             runs_per_iteration=10)

    labels = ['50 nodes', '100 nodes', '200 nodes']
    set_font_sizes()

    linestyles = ['dotted', 'dashed', 'solid']
    for i, _nrmses in enumerate(nrmses):
        plt.plot(density, np.squeeze(_nrmses), color='black',
                 marker='.', linestyle=linestyles[i], label=labels[i])

    maxlim = np.max(nrmses) + 0.05
    minlim = np.min(nrmses) - 0.05
    plt.ylim(minlim, maxlim)

    plt.ylabel('NRMSE')
    plt.xlabel('Input density')
    plt.legend(fancybox=False, loc='upper left', bbox_to_anchor=(0.0, 1.0))
    plt.hlines(y = np.arange(0.0, 1.05, 0.05), xmin=0.0, xmax=1.0,
               linewidth=0.2)

    maxlim = np.max(nrmses) + 0.15
    minlim = np.min(nrmses) - 0.05
    plt.ylim(minlim, maxlim)

    plt.margins(0.0)
    plt.savefig('plots/' + get_time())
    plt.show()


def grid_search_output_density(dataset):
    # NB: The keys will always be sorted for reproducibility, so keep them
    # sorted here.
    hidden_nodes = [50, 100, 200]
    density = np.arange(0.1, 1.1, 0.1)
    params = {
        'hidden_nodes': hidden_nodes,
        'w_out_density': density,
    }

    nrmses = evaluate_esn_2d(dataset, params,
                             evaluate_esn_output_density,
                             runs_per_iteration=10)

    labels = ['50 nodes', '100 nodes', '200 nodes']
    set_font_sizes()

    linestyles = ['dotted', 'dashed', 'solid']
    for i, _nrmses in enumerate(nrmses):
        plt.plot(density, np.squeeze(_nrmses), color='black',
                 marker='.', linestyle=linestyles[i], label=labels[i])

    maxlim = np.max(nrmses) + 0.05
    minlim = np.min(nrmses) - 0.05
    plt.ylim(minlim, maxlim)

    plt.ylabel('NRMSE')
    plt.xlabel('Output density')
    plt.legend(fancybox=False, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.hlines(y = np.arange(0.0, 2.05, 0.05), xmin=0.0, xmax=1.0,
               linewidth=0.2)

    maxlim = np.max(nrmses) + 0.15
    minlim = np.min(nrmses) - 0.05
    plt.ylim(minlim, maxlim)

    plt.margins(0.0)
    plt.savefig('plots/' + get_time())
    plt.show()


def grid_search_partial_visibility(dataset):
    # nrmses = pickle.load(open('tmp/partial_visibility', 'rb'))

    input_density = np.arange(0.0, 1.025, 0.025)
    output_density = np.arange(0.0, 1.025, 0.025)
    params = {
        'w_in_density': input_density,
        'w_out_density': output_density
    }

    nrmses = evaluate_esn_2d(dataset, params, eval_partial_visibility,
                             runs_per_iteration=10)
    pickle.dump(nrmses, open('tmp/' + get_time(), 'wb'))

    set_font_sizes()

    sns.heatmap(list(reversed(nrmses)), vmin=0.2, vmax=0.6, square=True)
    ax = plt.axes()

    # Fix half cells at the top and bottom. This is a current bug in Matplotlib.
    ax.set_ylim(ax.get_ylim()[0]+0.5, 0.0)

    x_width = ax.get_xlim()[1]
    y_width = ax.get_ylim()[0]

    plt.xticks([0.0, 0.5*x_width, x_width], [0.0, 0.5, 1.0])
    plt.yticks([0.0, 0.5*y_width, y_width], [1.0, 0.5, ''])

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    plt.xlabel('Output density')
    plt.ylabel('Input density')
    ax.collections[0].colorbar.set_label('NRMSE')

    plt.margins(0.0)
    plt.savefig('plots/' + get_time())
    plt.show()


def grid_search_input_scaling_input_distrib(dataset):
    # NB: The keys will always be sorted for reproducibility, so keep them
    # sorted here.
    distrib = [Distribution.gaussian, Distribution.uniform, Distribution.fixed]
    scaling = np.arange(0.1, 1.1, 0.1)
    params = {
        'input_scaling': scaling,
        'w_in_density': distrib,
    }

    nrmses = evaluate_esn_2d(dataset, params,
                             eval_input_distrib,
                             runs_per_iteration=10)

    # We need to transpose, since we want the input scaling to be the x-axis,
    # but it is before w_in_density alphabetically.
    nrmses = np.array(nrmses).T

    labels = ['gaussian', 'uniform', 'fixed']
    set_font_sizes()

    linestyles = ['dotted', 'dashed', 'solid']
    for i, _nrmses in enumerate(nrmses):
        plt.plot(scaling, np.squeeze(_nrmses), color='black',
                 marker='.', linestyle=linestyles[i], label=labels[i])

    maxlim = np.max(nrmses) + 0.05
    minlim = np.min(nrmses) - 0.05
    plt.ylim(minlim, maxlim)

    plt.ylabel('NRMSE')
    plt.xlabel('Input scaling')
    plt.legend(fancybox=False, loc='upper left', bbox_to_anchor=(0.0, 1.0))
    plt.hlines(y = np.arange(0.0, 1.05, 0.05), xmin=0.0, xmax=1.0,
               linewidth=0.2)

    maxlim = np.max(nrmses) + 0.15
    minlim = np.min(nrmses) - 0.05
    plt.ylim(minlim, maxlim)

    plt.margins(0.0)
    plt.savefig('plots/' + get_time())
    plt.show()


def grid_search_input_density_input_scaling(dataset):
    input_scaling = np.arange(0.0, 2.1, 0.1)
    input_density = np.arange(0.0, 1.05, 0.05)
    params = {
        'input_scaling': input_scaling,
        'w_in_density': input_density,
    }

    nrmses = evaluate_esn_2d(dataset, params,
                             evaluate_esn_input_density_scaling,
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

    plt.xlabel('Input density')
    plt.ylabel('Input scaling')
    ax.collections[0].colorbar.set_label('NRMSE')

    plt.show()


def input_noise(dataset):
    pass


def performance_sweep(dataset):
    hidden_nodes = [50, 100, 150, 200]
    params = { 'hidden_nodes': hidden_nodes }
    nrmses = evaluate_esn_1d(dataset, params, runs_per_iteration=10)

    set_font_sizes()

    plt.plot(hidden_nodes, nrmses, color='black', linestyle='dashed', marker='.')

    plt.ylabel('NARMA10 - NRMSE')
    plt.xlabel('Reservoir size')
    plt.xticks(np.arange(min(hidden_nodes), max(hidden_nodes) + 1, 50))

    maxlim = np.max(nrmses) + 0.05
    minlim = np.min(nrmses) - 0.05
    plt.ylim(minlim, maxlim)
    plt.hlines(y = np.arange(0.0, 1.05, 0.05), xmin=50, xmax=200, linewidth=0.2)

    plt.margins(0.0)
    plt.savefig('plots/' + get_time())
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

    plt.savefig('plots/' + get_time())
    plt.show()
