from datetime import datetime
from functools import wraps
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import matrix
import networkx as nx
import torch

import ESN
from ESN import Distribution
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
        plt.scatter(x, np.squeeze(_nrmses), marker=markers[i], label=labels[i])

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

def scatter_3d(G, title='', cols=None, show=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs = [], [], []
    for n in G.nodes:
        xs.append(G.nodes[n]['pos'][0])
        ys.append(G.nodes[n]['pos'][1])
        zs.append(G.nodes[n]['pos'][2])

    ax.set_title(title)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    if cols is None:
        c = 'black'
    else:
        c = cols

    ax.scatter(xs, ys, zs, c=c, cmap='binary')
    if show:
        plt.show()


def scatter_2d(G):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs = [], [], []
    for n in G.nodes:
        xs.append(G.nodes[n]['pos'][0])
        ys.append(G.nodes[n]['pos'][1])
        zs.append(G.nodes[n]['pos'][2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    ax.scatter(xs, ys, zs, color='black')
    plt.show()


def plot_trisurf(data, labels=None, ax=None, title=None, xlim=None, ylim=None,
                 zlim=None, azim=-45, elev=None, show=True, label=None, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')

    if title is not None:
        ax.set_title(title)

    if labels is not None:
        ax.set_xlabel(labels['x'], labelpad=9)
        ax.set_ylabel(labels['y'], labelpad=9)
        ax.set_zlabel(labels['z'], labelpad=9)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if zlim is not None:
        ax.set_zlim(zlim[0], zlim[1])

    vmin, vmax = min(data['z'])-0.05, max(data['z'])+0.05
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = 'gray'

    _ax = ax.plot_trisurf(data['x'], data['y'], data['z'], label=label,
                          cmap=cmap, norm=norm, **kwargs)
    ax.view_init(azim=azim, elev=elev)

    if label is not None:
        _ax._facecolors2d=_ax._facecolors3d
        _ax._edgecolors2d=_ax._edgecolors3d

    if show:
        plt.show()


def plot_lattice(G, title='', ax=None, neigh_color=False, edge_color=False,
                 cols=None, show=True, directed=False, cmap_r=False, alpha=1.0,
                 color_directed=False, hide_axes=False):
    pos = nx.get_node_attributes(G, 'pos')

    if ax is None:
        _, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_title(title)

    if neigh_color:
        A = nx.to_numpy_matrix(G)
        A = torch.FloatTensor(A).data.numpy()
        node_colors = A[0]
    elif cols is not None:
        node_colors = cols
    else:
        node_colors = 'black'

    if edge_color:
        edge_weights = [d['weight'] for u,v,d in G.edges(data=True)]
        edge_colors = ['red' if w == -1 else 'black' for w in edge_weights]
    elif color_directed:
        edges = set(G.edges())
        edge_colors = ['black' if (e[1], e[0]) not in edges else 'white' for e in G.edges()]
    else:
        edge_colors = 'black'

    cmap = 'binary_r' if cmap_r else 'binary'

    G = G if not directed else G.to_directed()
    nx.draw(G, pos=pos, ax=ax, with_labels=False, node_size=30,
            node_color=node_colors, edge_color=edge_colors, cmap=cmap,
            alpha=alpha)

    x1, x2, y1, y2 = ax.axis()
    ax.axis((x1-0.5, x2+0.5, y1-0.5, y2+0.5))

    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    if hide_axes:
        ax.set_axis_off()

    plt.tight_layout()
    if show:
        plt.show()


def plot_df(df, groupby, axes, labels=None):
    grouped_df = df.groupby(groupby).mean().reset_index()

    data = [grouped_df[axis] for axis in axes]
    plt.plot(*data)

    if labels is not None:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])

    plt.show()


def plot_df_trisurf(df, groupby, axes, agg=['mean'], show=True, title='', ax=None,
                    azim=None, elev=None, zlim=None, **kwargs):
    if ax is None:
        axs = get_3d_subplot_axs(len(agg))

    plt.suptitle(title)

    for i, _agg in enumerate(agg):
        if _agg == 'min':
            grouped_df = df.groupby(groupby).min().reset_index()
        elif _agg =='max':
            grouped_df = df.groupby(groupby).max().reset_index()
        else:
            grouped_df = df.groupby(groupby).mean().reset_index()

        data = {
            'x': grouped_df[axes[0]],
            'y': grouped_df[axes[1]],
            'z': grouped_df[axes[2]],
        }

        if 'labels' not in kwargs:
            kwargs['labels'] = {
                'x': axes[0],
                'y': axes[1],
                'z': axes[2],
            }

        if ax is None:
            cur_ax = axs[i]
        else:
            cur_ax = ax

        _azim = azim if type(azim) != list else azim[i]
        _elev = elev if type(elev) != list else elev[i]
        _zlim = zlim if type(zlim) != list else zlim[i]

        plot_trisurf(data=data, ax=cur_ax, show=False, title=title,
                     azim=_azim, elev=_elev, zlim=_zlim, **kwargs)

    if show:
        if 'label' in kwargs:
            ax.legend()
        plt.show()


def reject_outliers(data, m=2.0):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0
    return data[s<m]


def plot_vector_hist(vector, n_bins, m=20.0, ax=None, show=False, **kwargs):
    # Clip outliers.
    if m is not None:
        wres_outliers = reject_outliers(vector, m=m)
        min_clip, max_clip = np.min(wres_outliers), np.max(wres_outliers)
        np.clip(vector, min_clip, max_clip, out=vector)
    else:
        min_clip, max_clip = np.min(vector), np.max(vector)

    _plt = plt
    if ax is not None:
        _plt = ax

    bin_width = (max_clip - min_clip) / n_bins
    bins = np.arange(min_clip, max_clip + bin_width, bin_width)
    _plt.hist(vector, bins=bins, edgecolor='black', **kwargs)

    if show:
        plt.show()


def plot_esn_weight_hist(params, **kwargs):
    esn = ESN.ESN(**params)
    weights = esn.w_res.data.numpy().flatten()
    plot_vector_hist(weights, **kwargs)



def get_3d_subplot_axs(n):
    aspect_ratio = 0.4 if n > 1 else 1.0
    fig = plt.figure(figsize=plt.figaspect(aspect_ratio))
    axs = []

    for i in range(1, n+1):
        ax = plt.subplot(1, n, i, projection='3d')
        axs.append(ax)

    return axs


def get_figsize():
    return plt.rcParams["figure.figsize"]


def set_figsize(width, height):
    old_figsize = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = (width, height)
