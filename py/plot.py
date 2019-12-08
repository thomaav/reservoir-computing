from datetime import datetime
from functools import wraps
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

from ESN import Distribution
from gridsearch import evaluate_esn_1d, evaluate_esn_2d
from metric import *
from util import snr


def default_font_size(fn):
    def wrapped(*args, **kwargs):
        plt.rc('legend', fontsize=14)
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)
        plt.rc('axes', labelsize=16)
        fn(*args, **kwargs)
    return wrapped


def show(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        fn(*args, **kwargs)
        plt.margins(0.0)
        plt.savefig('plots/' + get_time())
        plt.show()
    return wrapped


def default_plot_settings(output, xmin, xmax):
    maxlim = np.max(output) + 0.05
    minlim = np.min(output) - 0.05
    plt.ylim(minlim, maxlim)
    plt.hlines(y = np.arange(0.0, 1.05, 0.05), xmin=xmin, xmax=xmax, linewidth=0.2)


def get_time():
    return datetime.now().strftime("%m-%d-%Y %H:%M:%S")


@default_font_size
@show
def plot_input_density(dataset):
    # NB: The keys will always be sorted for reproducibility, so keep them
    # sorted here.
    hidden_nodes = [50, 100, 200]
    density = np.arange(0.1, 1.1, 0.1)
    params = {
        'hidden_nodes': hidden_nodes,
        'w_in_density': density,
    }

    nrmses, stds = evaluate_esn_2d(dataset, params, runs_per_iteration=50)
    pickle.dump(nrmses, open('tmp/input_density_nrmse' + get_time(), 'wb'))
    pickle.dump(stds, open('tmp/input_density_std' + get_time(), 'wb'))

    # nrmses = pickle.load(open('tmp/input_density_nrmse', 'rb'))
    # stds = pickle.load(open('tmp/input_density_std', 'rb'))

    labels = ['50 nodes', '100 nodes', '200 nodes']
    linestyles = ['dotted', 'dashed', 'solid']
    for i, _nrmses in enumerate(nrmses):
        plt.errorbar(density, np.squeeze(_nrmses), yerr=stds[i], capsize=3.0,
                     color='black', marker='.', linestyle=linestyles[i], label=labels[i])

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


@default_font_size
@show
def plot_output_density(dataset):
    # NB: The keys will always be sorted for reproducibility, so keep them
    # sorted here.
    hidden_nodes = [50, 100, 200]
    density = np.arange(0.1, 1.1, 0.1)
    params = {
        'hidden_nodes': hidden_nodes,
        'w_out_density': density,
    }

    nrmses, stds = evaluate_esn_2d(dataset, params, runs_per_iteration=10)

    labels = ['50 nodes', '100 nodes', '200 nodes']
    linestyles = ['dotted', 'dashed', 'solid']
    for i, _nrmses in enumerate(nrmses):
        x = density*hidden_nodes[i]
        plt.plot(x, np.squeeze(_nrmses), color='black',
                 marker='.', linestyle=linestyles[i], label=labels[i])

    plt.ylabel('NRMSE')
    plt.xlabel('Output density')
    plt.legend(fancybox=False, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.hlines(y = np.arange(0.0, 2.05, 0.05), xmin=0.0, xmax=200.0, linewidth=0.2)

    maxlim = np.max(nrmses) + 0.15
    minlim = np.min(nrmses) - 0.05
    plt.ylim(minlim, maxlim)


@default_font_size
@show
def plot_output_nodes(dataset):
    hidden_nodes = [50, 100, 200]
    density = np.arange(0.05, 1.05, 0.05)
    params = {
        'hidden_nodes': hidden_nodes,
        'w_out_density': density,
    }

    nrmses = pickle.load(open('tmp/output_nodes', 'rb'))

    # nrmses, stds = evaluate_esn_2d(dataset, params, runs_per_iteration=10)
    # pickle.dump(nrmses, open('tmp/output_nodes-' + get_time(), 'wb'))

    labels = ['50 nodes', '100 nodes', '200 nodes']
    colors = ['red', 'green', 'blue']
    markers = ['.', '+', '^']
    for i, _nrmses in enumerate(nrmses):
        x = density*hidden_nodes[i]
        plt.scatter(x, np.squeeze(_nrmses), color='black', marker=markers[i], label=labels[i])

    plt.ylabel('NRMSE')
    plt.xlabel('Output nodes')
    plt.legend(fancybox=False, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.hlines(y = np.arange(0.0, 2.05, 0.05), xmin=0.0, xmax=125.0, linewidth=0.2)

    maxlim = 1.0
    minlim = np.min(nrmses) - 0.05
    plt.ylim(minlim, maxlim)

    maxlim = 125.0
    minlim = 0.0
    plt.xlim(minlim, maxlim)


@default_font_size
@show
def plot_partial_visibility(dataset):
    # nrmses = pickle.load(open('tmp/partial_visibility', 'rb'))

    input_density = np.arange(0.0, 1.025, 0.025)
    output_density = np.arange(0.0, 1.025, 0.025)
    params = {
        'w_in_density': input_density,
        'w_out_density': output_density
    }

    nrmses, stds = evaluate_esn_2d(dataset, params, runs_per_iteration=10)
    pickle.dump(nrmses, open('tmp/' + get_time(), 'wb'))

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


@default_font_size
@show
def plot_input_scaling_input_distrib(dataset):
    # NB: The keys will always be sorted for reproducibility, so keep them
    # sorted here.
    distrib = [Distribution.gaussian, Distribution.uniform, Distribution.fixed]
    scaling = np.arange(0.1, 1.1, 0.1)
    params = {
        'input_scaling': scaling,
        'w_in_distrib': distrib,
    }

    nrmses, stds = evaluate_esn_2d(dataset, params, runs_per_iteration=10)

    # We need to transpose, since we want the input scaling to be the x-axis,
    # but it is before w_in_density alphabetically.
    nrmses = np.array(nrmses).T

    labels = ['gaussian', 'uniform', 'fixed']
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


@default_font_size
@show
def plot_w_res_density_w_res_distrib(dataset):
    # NB: The keys will always be sorted for reproducibility, so keep them
    # sorted here.
    density = np.arange(0.1, 1.1, 0.1)
    distrib = [Distribution.gaussian, Distribution.uniform]
    params = {
        'w_res_density': density,
        'w_res_distrib': distrib,
    }

    nrmses, stds = evaluate_esn_2d(dataset, params, runs_per_iteration=10)

    # We need to transpose, since we want the input scaling to be the x-axis,
    # but it is before w_in_density alphabetically.
    nrmses = np.array(nrmses).T

    labels = ['gaussian', 'uniform', 'fixed']
    linestyles = ['dotted', 'dashed', 'solid']
    for i, _nrmses in enumerate(nrmses):
        plt.plot(density, np.squeeze(_nrmses), color='black',
                 marker='.', linestyle=linestyles[i], label=labels[i])

    plt.ylabel('NRMSE')
    plt.xlabel('Reservoir density')
    plt.legend(fancybox=False, loc='upper left', bbox_to_anchor=(0.0, 1.0))
    plt.hlines(y = np.arange(0.0, 1.05, 0.05), xmin=0.0, xmax=1.0,
               linewidth=0.2)

    maxlim = np.max(nrmses) + 0.15
    minlim = np.min(nrmses) - 0.05
    plt.ylim(minlim, maxlim)


@default_font_size
@show
def plot_input_noise(dataset):
    # Logspace from 0.001 to 0.14, as 0.14 is an SNR of ~0.0 with NARMA10.
    noise_std = np.logspace(-2.8239, -0.841, 50)
    params = { 'awgn_test_std': noise_std }
    test_snrs = []
    nrmses = evaluate_esn_1d(dataset, params, runs_per_iteration=10, test_snrs=test_snrs)

    plt.plot(test_snrs, nrmses, color='black', linestyle='dashed', marker='.')

    plt.ylabel('NARMA10 - NRMSE')
    plt.xlabel('Input signal to noise ratio')
    plt.xticks(np.arange(0, max(test_snrs) + 1, 5))

    maxlim = np.max(nrmses) + 0.05
    minlim = np.min(nrmses) - 0.05
    plt.ylim(minlim, maxlim)
    plt.hlines(y = np.arange(0.0, 2.0, 0.05), xmin=-5.0, xmax=max(test_snrs), linewidth=0.2)


@default_font_size
@show
def plot_input_noise_trained(dataset):
    # Logspace from 0.001 to 0.14, as 0.14 is an SNR of ~0.0 with NARMA10.
    test_noise_std = np.logspace(-2.8239, -0.841, 30)
    train_noise_std = np.logspace(-2.8239, -0.841, 30)
    params = {
        'awgn_test_std': test_noise_std,
        'awgn_train_std': train_noise_std,
    }

    u_train, y_train, u_test, y_test = dataset
    print(snr(u_train.var(), min(train_noise_std)**2))

    nrmses, stds = evaluate_esn_2d(dataset, params, runs_per_iteration=10)
    nrmses = np.array(nrmses).T

    sns.heatmap(list(reversed(nrmses)), vmin=0.0, vmax=1.0, square=True)
    ax = plt.axes()

    # Fix half cells at the top and bottom. This is a current bug in Matplotlib.
    ax.set_ylim(ax.get_ylim()[0]+0.5, 0.0)

    x_width = ax.get_xlim()[1]
    y_width = ax.get_ylim()[0]

    plt.xticks([0.0, 0.5*x_width, x_width], [40, 20, 0])
    plt.yticks([0.0, 0.5*y_width, y_width], [0, 20, 40])

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    plt.xlabel('Test signal to noise ratio')
    plt.ylabel('Train signal to noise ratio')
    ax.collections[0].colorbar.set_label('NRMSE')


@default_font_size
@show
def plot_adc_quantization(dataset):
    min_bits = 4
    max_bits = 14
    quantizations = np.array([2**n for n in range(min_bits, max_bits+2, 2)])
    hidden_nodes = [50, 100, 200, 400]
    params = {
        'adc_quantization': quantizations,
        'hidden_nodes': hidden_nodes,
    }

    # nrmses, stds = evaluate_esn_2d(dataset, params, runs_per_iteration=20)
    # nrmses = np.array(nrmses).T
    # stds = np.array(stds).T
    # pickle.dump(nrmses, open('tmp/adc_quantization_nrmse' + get_time(), 'wb'))
    # pickle.dump(stds, open('tmp/adc_quantization_std' + get_time(), 'wb'))

    nrmses = pickle.load(open('tmp/adc_quantization_nrmse', 'rb'))
    stds = pickle.load(open('tmp/adc_quantization_std', 'rb'))

    labels = ['50 nodes', '100 nodes', '200 nodes', '400 nodes']
    linestyles = ['dotted', 'dashed', 'solid', 'dashdot']
    for i, _nrmses in enumerate(nrmses):
        plt.errorbar(quantizations, np.squeeze(_nrmses), yerr=stds[i], capsize=3.0,
                     color='black', marker='.', linestyle=linestyles[i], label=labels[i])

    maxlim = np.max(nrmses) + 0.05
    minlim = np.min(nrmses) - 0.05
    plt.ylim(minlim, maxlim)

    maxlim = 2**max_bits + 2000.0
    minlim = 2**min_bits - 2.0
    plt.xlim(minlim, maxlim)
    plt.xscale('log', basex=2)
    plt.xticks(np.logspace(min_bits, max_bits, base=2, num=6), np.arange(min_bits, max_bits+2, 2))

    plt.ylabel('NRMSE')
    plt.xlabel('Quantization bits for output')
    plt.legend(fancybox=False, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.hlines(y = np.arange(0.0, 1.05, 0.05), xmin=0.0, xmax=2**(max_bits+1), linewidth=0.2)
    plt.tight_layout()


@default_font_size
@show
def performance_sweep(dataset):
    hidden_nodes = [50, 100, 150, 200]
    params = { 'hidden_nodes': hidden_nodes }
    nrmses = evaluate_esn_1d(dataset, params, runs_per_iteration=10)

    plt.plot(hidden_nodes, nrmses, color='black', linestyle='dashed', marker='.')

    plt.ylabel('NARMA10 - NRMSE')
    plt.xlabel('Reservoir size')
    plt.xticks(np.arange(min(hidden_nodes), max(hidden_nodes) + 1, 50))

    default_plot_settings(nrmses, 50, 200)


@show
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
