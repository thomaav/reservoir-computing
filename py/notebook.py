import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import torch
import networkx as nx
from collections import OrderedDict

import dataset as ds
from ESN import ESN, Distribution, from_square_G, create_delay_line
from metric import esn_nrmse, evaluate_esn, esn_mc
from gridsearch import experiment, load_experiment
from plot import plot_df_trisurf, set_figsize, get_figsize, plot_lattice, plot_vector_hist
from matrix import euclidean, inv, inv_squared, inv_cubed
from experiment import find_best_node_to_add

default_w, default_h = get_figsize()
FIG_DIR = '../master/figures/'


# GENERAL.


def esn_general_performance():
    params = OrderedDict()
    params['hidden_nodes'] = np.arange(20, 660, 10)
    params['w_res_density'] = [0.1]
    params['readout'] = ['rr']

    df = experiment(esn_nrmse, params, runs=20)
    df.to_pickle('experiments/esn_general_performance.pkl')
    pass


def default_plot_settings():
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('axes', labelsize=16)


def save_plot(f):
    plt.savefig(FIG_DIR + f, dpi=200)


# BACKGROUND: Background.


def plot_NARMA10(range=[0, -1]):
    ax = plt.gca()

    _, y_train, _, _ = ds.dataset
    i, j = range[0], range[1]
    y = y_train[i:j]

    plt.plot(y, color='black')

    default_plot_settings()
    plt.margins(0.0)

    plt.xlabel('Time step', labelpad=10)
    plt.ylabel('NARMA10 output', labelpad=10)

    plt.ylim((0.1, 0.7))
    plt.hlines(ax.get_yticks(), ax.get_xlim()[0], ax.get_xlim()[1], linewidth=0.2)

    plt.tight_layout()
    save_plot('NARMA10.png')
    plt.show()


def plot_NARMA_nonlinearity():
    xs = []
    ys = []
    xys = []
    for x in np.linspace(0.0, 0.5, 10):
        for y in np.linspace(0.0, 0.5, 10):
            xs.append(x)
            ys.append(y)
            xys.append(x*y)

    ax = plt.gca(projection='3d')
    norm = matplotlib.colors.Normalize(vmin=-0.05, vmax=0.35)
    ax.plot_trisurf(xs, ys, xys, cmap='gray', norm=norm)

    #todo: remove diagonals

    ax.view_init(azim=200, elev=30)
    ax.set_zlim(min(xys), max(xys))
    plt.margins(0.0)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.set_xticks([min(xs), max(xs)])
    ax.set_yticks([min(ys), max(ys)])
    ax.set_zticks([min(xys), max(xys)])

    ax.set_xlabel(r'y', fontsize=18, labelpad=-5)
    ax.set_ylabel(r'x', fontsize=18, labelpad=-5)
    zlabel = ax.set_zlabel(r'x $\cdot$ y', rotation=90, fontsize=18, labelpad=-10)

    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)

    for x in ax.xaxis.get_major_ticks(): x.label.set_fontsize(14)
    for y in ax.yaxis.get_major_ticks(): y.label.set_fontsize(14)
    for z in ax.zaxis.get_major_ticks(): z.label.set_fontsize(14)

    plt.tight_layout()
    save_plot('NARMA-nonlinearity.png')
    plt.show()


# EXPERIMENTS: Random Geometric Graphs.


def plot_rgg_example(save=True):
    from matrix import waxman
    from plot import scatter_3d

    G = waxman(200, alpha=1.0, beta=1.0)

    xs, ys = [], []
    for n in G.nodes:
        xs.append(G.nodes[n]['pos'][0])
        ys.append(G.nodes[n]['pos'][1])

    ax = plt.gca()

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])

    ax.scatter(xs, ys, color='black')

    plt.tight_layout()
    if save:
        save_plot('RGG-example.png')
    plt.show()


def rgg_volume_size():
    from metric import memory_capacity, evaluate_esn

    params = OrderedDict()
    params['w_res_type'] = ['waxman']
    params['hidden_nodes'] = [100]
    params['spectral_radius'] = [None]
    params['dist_function'] = [inv, inv_squared]
    params['dim_size'] = [1] + list(np.arange(10, 510, 10))

    df = experiment(esn_nrmse, params, runs=10, esn_attributes=['org_spectral_radius'])
    df.to_pickle('experiments/rgg_volume_size.pkl')


def plot_rgg_volume_size():
    df = load_experiment('experiments/rgg_volume_size.pkl')

    df['dist_function'] = df['dist_function'].apply(
        lambda f: f.__name__ if not isinstance(f, str) else f
    )

    inv_df = df.loc[df['dist_function'] == inv.__name__]
    inv_squared_df = df.loc[df['dist_function'] == inv_squared.__name__]

    inv_df = inv_df.groupby(['dim_size']).min().reset_index()
    inv_squared_df = inv_squared_df.groupby(['dim_size']).min().reset_index()

    file_names = ['RGG-volume-size-inv.png', 'RGG-volume-size-inv-squared.png']
    for i, df in enumerate([inv_df, inv_squared_df]):
        default_plot_settings()
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        ax1.plot(df['dim_size'], df['esn_nrmse'], color='black', linestyle='solid', label='NRMSE')
        ax2.plot(df['dim_size'], df['org_spectral_radius'], color='black', linestyle='dashed', label='Spectral radius')
        ax2.plot(df['dim_size'], [1]*len(df['dim_size']), color='grey', linestyle='dotted')

        ax1.set_xlabel('Volume size')
        ax1.set_ylabel('NARMA-10 NRMSE')
        ax2.set_xlabel('Volume size')
        ax2.set_ylabel('Spectral radius')

        if df.equals(inv_df):
            ax2.set_ylim(0.0, 2.0)
        elif df.equals(inv_squared_df):
            ax1.set_ylim(0.0, 1.0)
            ax1.set_xlim(0, 150)
            ax2.set_ylim(0.0, 2.0)
            ax2.set_xlim(0, 150)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
        plt.legend(lines + lines2, labels + labels2, bbox_to_anchor=(0., 1.02, 1., .102),
                   loc='lower left', ncol=2, mode="expand", borderaxespad=0., fancybox=False)

        plt.tight_layout()
        save_plot(file_names[i])
        plt.show()


def rgg_dist_performance():
    params = OrderedDict()
    params['w_res_type'] = ['waxman']
    params['hidden_nodes'] = np.arange(20, 90, 10)
    params['dist_function'] = [euclidean, inv, inv_squared, inv_cubed]
    params['readout'] = ['rr']

    df = experiment(esn_nrmse, params, runs=20)
    df.to_pickle('experiments/rgg_dist_performance.pkl')


def plot_rgg_dist_performance(agg='mean', file_name=None):
    df = load_experiment('experiments/rgg_dist_performance.pkl')
    esn = load_experiment('experiments/esn_general_performance.pkl')
    esn['hidden_nodes'] = esn[esn['hidden_nodes'] <= 80]['hidden_nodes']

    df['dist_function'] = df['dist_function'].apply(
        lambda f: f.__name__ if not isinstance(f, str) else f
    )

    inv_perf = df.loc[df['dist_function'] == inv.__name__]
    inv_squared_perf = df.loc[df['dist_function'] == inv_squared.__name__]
    inv_cubed_perf = df.loc[df['dist_function'] == inv_cubed.__name__]

    labels = ['ESN', '1/d', '1/d^2', '1/d^3']
    linestyles = ['dashdot', 'dotted', 'dashed', 'solid']

    default_plot_settings()
    for i, df in enumerate([esn, inv_perf, inv_squared_perf, inv_cubed_perf]):
        if agg == 'mean':
            grouped_df = df.groupby(['hidden_nodes']).mean().reset_index()
        elif agg == 'min':
            grouped_df = df.groupby(['hidden_nodes']).min().reset_index()

        plt.plot(grouped_df['hidden_nodes'], grouped_df['esn_nrmse'], label=labels[i],
                 color='black', linestyle=linestyles[i])

    plt.xlabel('Hidden nodes')
    plt.ylabel('NARMA-10 NRMSE')

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2,
               mode="expand", borderaxespad=0., fancybox=False)

    plt.tight_layout()
    if file_name is not None:
        save_plot(file_name)
    plt.show()


def rgg_dist_mc():
    params = OrderedDict()
    params['w_res_type'] = ['waxman']
    params['hidden_nodes'] = [40, 80, 150]
    params['dist_function'] = [inv_squared]
    params['input_scaling'] = np.arange(0.1, 1.6, 0.1)

    df = experiment(esn_mc, params, runs=10)
    df.to_pickle('experiments/rgg_dist_mc.pkl')


def plot_rgg_dist_mc():
    df = load_experiment('experiments/rgg_dist_mc.pkl')

    df40 = df.loc[df['hidden_nodes'] == 40]
    df80 = df.loc[df['hidden_nodes'] == 80]
    df150 = df.loc[df['hidden_nodes'] == 150]

    linestyles = ['solid', 'dashed', 'dotted']
    labels = ['40 nodes', '80 nodes', '150 nodes']
    for i, df in enumerate([df40, df80, df150]):
        grouped_df = df.groupby(['input_scaling', 'hidden_nodes']).max().reset_index()
        plt.plot(grouped_df['input_scaling'], grouped_df['esn_mc'], color='black',
                 linestyle=linestyles[i], label=labels[i])

    plt.xlabel('Input scaling')
    plt.ylabel('Short term memory capacity')

    plt.legend(fancybox=False)
    plt.tight_layout()
    save_plot('RGG-dist-mc.png')
    plt.show()


def rgg_dist_performance_is():
    params = OrderedDict()
    params['w_res_type'] = ['waxman']
    params['hidden_nodes'] = [40, 80, 150]
    params['dist_function'] = [inv_squared]
    params['input_scaling'] = np.arange(0.1, 1.6, 0.1)

    df = experiment(esn_nrmse, params, runs=10)
    df.to_pickle('experiments/rgg_dist_performance_is.pkl')


def plot_rgg_dist_performance_is():
    df = load_experiment('experiments/rgg_dist_performance_is.pkl')

    df40 = df.loc[df['hidden_nodes'] == 40]
    df80 = df.loc[df['hidden_nodes'] == 80]
    df150 = df.loc[df['hidden_nodes'] == 150]

    linestyles = ['solid', 'dashed', 'dotted']
    labels = ['40 nodes', '80 nodes', '150 nodes']
    for i, df in enumerate([df40, df80, df150]):
        grouped_df = df.groupby(['input_scaling', 'hidden_nodes']).min().reset_index()
        plt.plot(grouped_df['input_scaling'], grouped_df['esn_nrmse'], color='black',
                 linestyle=linestyles[i], label=labels[i])

    plt.xlabel('Input scaling')
    plt.ylabel('NARMA-10 NRMSE')

    plt.legend(fancybox=False)
    plt.tight_layout()
    save_plot('RGG-dist-performance-is.png')
    plt.show()


# EXPERIMENTS: Regular Tilings.


def plot_regular_tilings(save=False):
    from ESN import ESN
    from plot import plot_lattice

    esn_square = ESN(hidden_nodes=25, w_res_type='tetragonal')
    esn_hex = ESN(hidden_nodes=9, w_res_type='hexagonal')
    esn_tri = ESN(hidden_nodes=16, w_res_type='triangular')

    G_square = esn_square.G
    G_hex = esn_hex.G
    G_tri = esn_tri.G

    file_names = ['square.png', 'hex.png', 'triangular.png']
    for i, G in enumerate([G_square, G_hex, G_tri]):
        plot_lattice(G, hide_axes=True, show=False)
        if save:
            save_plot(file_names[i])
        plt.show()


def regular_tilings_performance():
    params = OrderedDict()
    params['hidden_nodes'] = [n*n for n in range(3, 21)]
    params['w_res_type'] = ['tetragonal', 'hexagonal', 'triangular']
    nrmse_df = experiment(esn_nrmse, params)
    nrmse_df.to_pickle('experiments/lattice_nrmse.pkl')


def plot_regular_tilings_performance():
    nrmse_df = load_experiment('experiments/lattice_nrmse.pkl')
    esn = load_experiment('experiments/esn_general_performance.pkl')
    esn['hidden_nodes'] = esn[esn['hidden_nodes'] <= 400]['hidden_nodes']

    grouped_df = nrmse_df.groupby(['hidden_nodes', 'w_res_type']).mean().reset_index()
    esn = esn.groupby(['hidden_nodes']).mean().reset_index()

    sq = grouped_df.loc[grouped_df['w_res_type'] == 'tetragonal']
    hex = grouped_df.loc[grouped_df['w_res_type'] == 'hexagonal']
    tri = grouped_df.loc[grouped_df['w_res_type'] == 'triangular']

    plt.plot(sq['hidden_nodes'], sq['esn_nrmse'], label='Square', color='black', linestyle='solid')
    plt.plot(hex['hidden_nodes'], hex['esn_nrmse'], label='Hexagonal', color='black', linestyle='dashed')
    plt.plot(tri['hidden_nodes'], tri['esn_nrmse'], label='Triangular', color='black', linestyle='dotted')
    plt.plot(esn['hidden_nodes'], esn['esn_nrmse'], label='ESN', color='black', linestyle='dashdot')

    plt.legend(fancybox=False, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.ylabel('NARMA-10 NRMSE')
    plt.xlabel('Hidden nodes')

    plt.tight_layout()
    save_plot('regular-tilings-performance.png')
    plt.show()


def regular_tilings_performance_is():
    params = OrderedDict()
    params['w_res_type'] = ['tetragonal', 'hexagonal', 'triangular']
    params['hidden_nodes'] = [n*n for n in range(5, 16)]
    params['input_scaling'] = np.arange(0.1, 1.6, 0.1)
    nrmse_df = experiment(esn_nrmse, params)
    nrmse_df.to_pickle('experiments/rt_performance_is.pkl')


def plot_regular_tilings_performance_is():
    df = load_experiment('experiments/rt_performance_is.pkl')

    sq = df.loc[df['w_res_type'] == 'tetragonal']
    hex = df.loc[df['w_res_type'] == 'hexagonal']
    tri = df.loc[df['w_res_type'] == 'triangular']

    groupby = ['hidden_nodes', 'input_scaling']
    axes    = ['hidden_nodes', 'input_scaling', 'esn_nrmse']
    agg     = ['mean']
    labels  = {'x': 'Hidden nodes', 'y': 'Input scaling', 'z': 'NARMA-10 NRMSE'}
    zlim    = (0.3, 0.65)
    xlim    = (min(sq['hidden_nodes']), max(sq['hidden_nodes']))
    ylim    = (min(sq['input_scaling']), max(sq['input_scaling']))
    azim    = -45
    elev    = 20

    file_names = ['sq', 'hex', 'tri']
    for i, df in enumerate([sq, hex, tri]):
        plot_df_trisurf(df=df, groupby=groupby, axes=axes, azim=azim, elev=elev, agg=agg,
                        zlim=zlim, show=False, labels=labels, xlim=xlim, ylim=ylim)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.10, right=0.9)
        save_plot(f'regular-tilings-performance-is-{file_names[i]}.png')
        plt.show()


def plot_directed_lattice(save=False):
    esn = ESN(hidden_nodes=25, w_res_type='tetragonal', dir_frac=0.25)
    plot_lattice(esn.G.reverse(), color_directed=True, show=False)
    ax = plt.gca()
    ax.set_axis_off()
    if save:
        save_plot('dir_lattice_025.png')
    plt.show()

    esn = ESN(hidden_nodes=25, w_res_type='tetragonal', dir_frac=0.75)
    plot_lattice(esn.G.reverse(), color_directed=True, show=False)
    ax = plt.gca()
    ax.set_axis_off()
    if save:
        save_plot('dir_lattice_075.png')
    plt.show()


def directed_lattice_performance():
    params = OrderedDict()
    params['w_res_type'] = ['tetragonal']
    params['hidden_nodes'] = [n*n for n in range(5, 26)]
    params['input_scaling'] = [0.1]
    params['w_in_distrib'] = [Distribution.fixed]
    params['dir_frac'] = [1.0]
    params['w_in_density'] = [1.0, 0.5]

    df = experiment(esn_nrmse, params, runs=20)
    df.to_pickle('experiments/directed_lattice_performance.pkl')


def plot_directed_lattice_performance():
    df = load_experiment('experiments/directed_lattice_performance.pkl')
    esn_df = load_experiment('experiments/esn_general_performance.pkl')

    dense_df = df.loc[df['w_in_density'] == 1.0]
    sparse_df = df.loc[df['w_in_density'] == 0.5]

    dense_grouped_df = dense_df.groupby(['hidden_nodes']).mean().reset_index()
    sparse_grouped_df = sparse_df.groupby(['hidden_nodes']).mean().reset_index()
    esn_grouped_df = esn_df.groupby(['hidden_nodes']).mean().reset_index()

    plt.xlabel('Hidden nodes')
    plt.ylabel('NARMA-10 NRMSE')

    plt.plot(dense_grouped_df['hidden_nodes'], dense_grouped_df['esn_nrmse'], label='Square (dense input)', color='black', linestyle='solid')
    plt.plot(sparse_grouped_df['hidden_nodes'], sparse_grouped_df['esn_nrmse'], label='Square (sparse input)', color='black', linestyle='dashed')
    plt.plot(esn_grouped_df['hidden_nodes'], esn_grouped_df['esn_nrmse'], label='ESN', color='black', linestyle='dashdot')

    plt.legend()
    plt.tight_layout()
    save_plot('rt-performance-big.png')
    plt.show()


def directed_regular_tilings_performance():
    params = OrderedDict()
    params['input_scaling'] = [0.1]
    params['w_res_type'] = ['tetragonal', 'hexagonal', 'triangular']
    params['hidden_nodes'] = [25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225]
    params['dir_frac'] = np.arange(0.0, 1.1, 0.1)
    lattice_dir_df = experiment(esn_nrmse, params, runs=10)
    lattice_dir_df.to_pickle('experiments/lattice_dir.pkl')


def plot_directed_regular_tilings_performance():
    lattice_dir_df = load_experiment('experiments/lattice_dir.pkl')
    esn_df = load_experiment('experiments/esn_general_performance.pkl')

    sq = lattice_dir_df.loc[lattice_dir_df['w_res_type'] == 'tetragonal']
    hex = lattice_dir_df.loc[lattice_dir_df['w_res_type'] == 'hexagonal']
    tri = lattice_dir_df.loc[lattice_dir_df['w_res_type'] == 'triangular']

    groupby = ['hidden_nodes', 'dir_frac']
    axes    = ['hidden_nodes', 'dir_frac', 'esn_nrmse']
    agg     = ['mean']
    zlims   = [(0.25, 0.6), (0.28, 0.6), (0.28, 0.45)]
    azim    = 40
    labels  = {'x': 'Hidden nodes', 'y': 'Directed edges', 'z': 'NARMA-10 NRMSE'}

    file_names = ['rt-dir-perf-sq.png', 'rt-dir-perf-hex.png', 'rt-dir-perf-tri.png']
    for i, df in enumerate([sq, hex, tri]):
        plot_df_trisurf(df=df, groupby=groupby, axes=axes, azim=azim, agg=agg,
                        zlim=zlims[i], labels=labels, show=False)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12, left=0.1)

        ax = plt.gca()
        ax.set_zlabel(ax.get_zlabel(), rotation=90)
        ax.zaxis.set_rotate_label(False)

        save_plot(file_names[i])
        plt.show()

    sq = sq.loc[sq['dir_frac'] == 1.0]
    sq = sq.groupby(['hidden_nodes', 'dir_frac']).mean().reset_index()
    hex = hex.loc[hex['dir_frac'] == 1.0]
    hex = hex.groupby(['hidden_nodes', 'dir_frac']).mean().reset_index()
    tri = tri.loc[tri['dir_frac'] == 1.0]
    tri = tri.groupby(['hidden_nodes', 'dir_frac']).mean().reset_index()
    esn = esn_df.groupby(['hidden_nodes']).mean().reset_index()
    esn = esn.loc[esn['hidden_nodes'] <= 230]

    plt.plot(sq['hidden_nodes'], sq['esn_nrmse'], color='black', label='Square', linestyle='solid')
    plt.plot(hex['hidden_nodes'], hex['esn_nrmse'], color='black', label='Hexagonal', linestyle='dashed')
    plt.plot(tri['hidden_nodes'], tri['esn_nrmse'], color='black', label='Triagonal', linestyle='dotted')
    plt.plot(esn['hidden_nodes'], esn['esn_nrmse'], color='black', label='ESN', linestyle='dashdot')

    plt.legend(fancybox=False, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.ylabel('NARMA-10 NRMSE')
    plt.xlabel('Hidden nodes')

    plt.xlim(0, max(sq['hidden_nodes'])+15)

    plt.tight_layout()
    save_plot('rt-dir-perf.png')
    plt.show()


def global_input_scheme_performance():
    from ESN import Distribution

    params = OrderedDict()
    params['hidden_nodes'] = 25*25
    params['w_res_density'] = 0.2
    def_esn = ESN(**params)

    params['w_res_type'] = 'tetragonal'
    params['hidden_nodes'] = 25*25
    params['w_in_distrib'] = Distribution.fixed
    params['input_scaling'] = 0.1
    params['dir_frac'] = 1.0
    dir_esn = ESN(**params)

    def_nrmse = evaluate_esn(ds.dataset, def_esn)
    dir_nrmse = evaluate_esn(ds.dataset, dir_esn)

    print('Hidden nodes:')
    print(' default:', def_esn.hidden_nodes)
    print(' lattice:', dir_esn.hidden_nodes)

    print()
    print('NRMSE:')
    print(' default:', def_nrmse)
    print(' lattice:', dir_nrmse)

    print()
    print('Lattice weights:')
    print(f' w_in:  unique-{np.unique(dir_esn.w_in)}')
    print(f' w_res: unique-{np.unique(dir_esn.w_res)}')


def unique_weights():
    from ESN import Distribution

    params = OrderedDict()
    params['w_in_distrib'] = [Distribution.fixed]
    params['input_scaling'] = [0.1]
    params['w_res_type'] = ['tetragonal']
    params['hidden_nodes'] = [100, 225, 400]
    params['dir_frac'] = [1.0]
    df = experiment(esn_nrmse, params, runs=10, esn_attributes=['w_in', 'w_res'])
    df['uwin'] = df['w_in'].apply(lambda m: len(np.unique(m[np.nonzero(m.data.numpy())])))
    df['uwres'] = df['w_res'].apply(lambda m: len(np.unique(m[np.nonzero(m.data.numpy())])))
    del df['w_in']
    del df['w_res']
    df.to_pickle('experiments/unique_weights_square.pkl')

    params = OrderedDict()
    params['hidden_nodes'] = [100, 225, 400]
    params['w_res_density'] = [0.1]
    df = experiment(esn_nrmse, params, runs=10, esn_attributes=['w_in', 'w_res'])
    df['uwin'] = df['w_in'].apply(lambda m: len(np.unique(m[np.nonzero(m.data.numpy())])))
    df['uwres'] = df['w_res'].apply(lambda m: len(np.unique(m[np.nonzero(m.data.numpy())])))
    del df['w_in']
    del df['w_res']
    df.to_pickle('experiments/unique_weights_esn.pkl')


def print_unique_weights():
    sq_df = load_experiment('experiments/unique_weights_square.pkl')
    esn_df = load_experiment('experiments/unique_weights_esn.pkl')

    agg = {
        'esn_nrmse': ['mean', 'std'],
        'uwin': ['mean', 'std'],
        'uwres': ['mean', 'std'],
    }

    sq_df = sq_df.groupby(['hidden_nodes']).agg(agg).reset_index()
    esn_df = esn_df.groupby(['hidden_nodes']).agg(agg).reset_index()

    print(sq_df)
    print(esn_df)


def plot_global_input_activations(save=False):
    from ESN import Distribution

    hidden_nodes = 144

    params = OrderedDict()
    params['hidden_nodes'] = hidden_nodes
    params['w_res_density'] = 0.1
    def_esn = ESN(**params)

    params = OrderedDict()
    params['w_in_distrib'] = Distribution.fixed
    params['input_scaling'] = 0.1
    params['w_res_type'] = 'tetragonal'
    params['hidden_nodes'] = hidden_nodes
    params['dir_frac'] = 1.0
    dir_esn = ESN(**params)

    def_nrmse = evaluate_esn(ds.dataset, def_esn)
    dir_nrmse = evaluate_esn(ds.dataset, dir_esn)

    l = 50
    n = 10

    file_names = ['esn-activations.png', 'sq-activations.png']
    for i, esn in enumerate([def_esn, dir_esn]):
        plt.xlabel('Time step')
        plt.ylabel(r'$\tanh$ activation')

        plt.plot(esn.X[esn.washout:esn.washout+l, :n], color='black')
        plt.tight_layout()
        if save:
            save_plot(file_names[i])
        plt.show()


def node_removal_impact():
    import copy

    params = OrderedDict()
    params['w_in_distrib'] = Distribution.fixed
    params['input_scaling'] = 0.1
    params['w_res_type'] = 'tetragonal'
    params['hidden_nodes'] = 144
    params['dir_frac'] = 1.0
    dir_esn = ESN(**params)

    def_nrmse = evaluate_esn(ds.dataset, dir_esn)
    nrmse_diffs = []

    for i in range(len(dir_esn.G.nodes)):
        esn = copy.deepcopy(dir_esn)
        esn.remove_hidden_node(i)
        nrmse_diffs.append(evaluate_esn(ds.dataset, esn) - def_nrmse)

    pickle.dump(nrmse_diffs, open('experiments/dir_diff_nrmses.pkl', 'wb'))


def plot_node_removal_impact():
    nrmse_diffs = np.array(pickle.load(open('experiments/dir_diff_nrmses.pkl', 'rb')))
    set_figsize(6.4, 4.8)
    plot_vector_hist(nrmse_diffs, n_bins=20, m=None, ax=plt.gca(), show=False)
    plt.xlabel('NARMA-10 NRMSE change')
    plt.ylabel('Frequency')
    plt.tight_layout()
    save_plot('removal-hist.png')
    plt.show()


def remove_nodes_performance():
    lattice_nrmses = pickle.load(open('experiments/deg_lattice_nrmses.pkl', 'rb'))
    lattices = pickle.load(open('experiments/deg_lattices.pkl', 'rb'))
    esn = load_experiment('experiments/esn_general_performance.pkl')

    esn = esn.loc[esn['hidden_nodes'] <= 140]
    esn = esn.groupby(['hidden_nodes']).mean().reset_index()
    def_nrmses = esn['esn_nrmse']
    n_nodes = esn['hidden_nodes']

    max_nodes = len(lattices)
    print(f'Lattice min NRMSE: {max_nodes - np.argmin(lattice_nrmses)} nodes with NRMSE {min(lattice_nrmses)}')
    print(f'Default min NRMSE: {max_nodes - np.argmin(def_nrmses)} nodes with NRMSE {min(def_nrmses)}')

    x = list(range(max_nodes, 0, -1))

    plt.title('NRMSE as we incrementally remove lattice nodes vs. mean random ESN')
    plt.plot(x, lattice_nrmses, label='Lattice')
    plt.plot(n_nodes, def_nrmses, label='ESN')

    plt.gca().invert_xaxis()
    plt.ylim((0.0, 1.0))

    plt.xlabel('Hidden nodes')
    plt.ylabel('NRMSE')

    plt.legend()
    plt.show()


def general_esn_shrink():
    params = OrderedDict()
    params['hidden_nodes'] = [1, 3] + list(range(5, 150, 5))
    params['w_res_density'] = [0.1]
    lattice_dir_df = experiment(esn_nrmse, params, runs=10)
    lattice_dir_df.to_pickle('experiments/esn_general_short.pkl')


def remove_esn_nodes_performance():
    esn_shrink_nrmses = pickle.load(open('experiments/esn_removed_nodes_nrmses.pkl', 'rb'))
    lattice_shrink_nrmses = pickle.load(open('experiments/deg_lattice_nrmses.pkl', 'rb'))

    esn_general = load_experiment('experiments/esn_general_short.pkl')
    esn_general = esn_general.groupby(['hidden_nodes']).mean().reset_index()

    max_nodes = len(lattice_shrink_nrmses)

    x = list(range(len(esn_shrink_nrmses), 0, -1))
    plt.plot(esn_general['hidden_nodes'], esn_general['esn_nrmse'], label='Random ESN', color='black', linestyle='dashed')
    plt.plot(x, esn_shrink_nrmses, label='ESN (shrinking)', color='black', linestyle='dotted')
    plt.plot(x, lattice_shrink_nrmses, label='Square grid', color='black', linestyle='solid')

    plt.gca().invert_xaxis()
    plt.ylim((0.0, 1.0))

    plt.xlabel('Hidden nodes remaining')
    plt.ylabel('NARMA-10 NRMSE')

    plt.legend()
    plt.tight_layout()
    save_plot('shrink-performance.png')
    plt.show()


def plot_node_removal(save=False):
    dir_esn = pickle.load(open('models/dir_esn.pkl', 'rb'))
    lattice_nrmses = pickle.load(open('experiments/deg_lattice_nrmses.pkl', 'rb'))
    lattices = pickle.load(open('experiments/deg_lattices.pkl', 'rb'))

    max_nodes = len(lattices)

    for i in [130, 70, 35, 20]:
        title = f'Lattice, {i} nodes, NRMSE {lattice_nrmses[max_nodes-i]:.3f}'
        plot_lattice(dir_esn.G.reverse(), edge_color='0.7', cols='0.7', show=False, ax=plt.gca())
        plot_lattice(lattices[max_nodes - i].reverse(), ax=plt.gca(), show=False)
        plt.gca().set_axis_off()
        plt.tight_layout()
        if save:
            save_plot(f'sq-grid-{i}.png')
        plt.show()


def plot_esn_node_removal():
    esns = pickle.load(open('experiments/esn_removed_nodes_esns.pkl', 'rb'))
    esn_nrmses = pickle.load(open('experiments/esn_removed_nodes_nrmses.pkl', 'rb'))

    linear_esn = esns[len(esns)-31]
    G = nx.from_numpy_matrix(linear_esn.numpy())
    nx.draw(G, pos=nx.spring_layout(G))
    plt.show()


def plot_growth(save=False):
    # Lattices.
    lattices = pickle.load(open('experiments/grow_actual_lattices.pkl', 'rb'))
    ds.dataset = pickle.load(open('dataset/ds_narma_grow.pkl', 'rb'))

    set_figsize(10, 6)

    for i in [0, 50, -1]:
        lattice = lattices[i]
        esn = from_square_G(lattice)
        nrmse = evaluate_esn(ds.dataset, esn)
        print(f'{len(lattice.nodes)} hidden nodes, NRMSE: {nrmse}')
        plot_lattice(lattice, show=False)

        plt.gca().set_axis_off()
        plt.tight_layout()
        if save:
            save_plot(f'sq-grid-grow-{len(lattice.nodes)}.png')
        plt.show()

    set_figsize(default_w, default_h)


def plot_growth_performance():
    set_figsize(6.4, 4.8)
    nrmses = pickle.load(open('experiments/grow_mid_nrmses.pkl', 'rb'))

    plt.xlabel('Hidden nodes')
    plt.ylabel('NARMA-10 NRMSE')

    plt.plot(range(74, 74+len(nrmses)), nrmses, color='black')
    plt.tight_layout()
    save_plot('grow-performance.png')
    plt.show()


def making_edges_undirected_performance():
    import copy
    from ESN import find_esn, Distribution
    from experiment import make_undirected_incrementally
    from experiment import evaluate_incremental_undirection

    changed_edges_file = 'experiments/changed_edges.pkl'

    # Remove edges.
    params = OrderedDict()
    params['w_res_type'] = 'tetragonal'
    params['hidden_nodes'] = 144
    params['dir_frac'] = 1.0
    params['input_scaling'] = 0.1
    params['w_in_distrib'] = Distribution.fixed
    esn = find_esn(dataset=ds.dataset, required_nrmse=0.26, **params)
    make_undirected_incrementally(ds.dataset, copy.deepcopy(esn), changed_edges_file)

    # Evaluate removals.
    nrmses, esns = evaluate_incremental_undirection(ds.dataset, esn, changed_edges_file, esns=True)
    lattices = [esn.G for esn in esns]
    pickle.dump(nrmses, open('experiments/changed_edges_nrmses.pkl', 'wb'))
    pickle.dump(lattices, open('experiments/changed_edges_lattices.pkl', 'wb'))


def plot_making_edges_undirected_performance():
    nrmses = pickle.load(open('experiments/changed_edges_nrmses.pkl', 'rb'))

    edges = len(nrmses)
    print(f'To undirected min NRMSE: {np.argmin(nrmses)} edges removed with NRMSE {min(nrmses)}')

    plt.plot([n/264 for n in range(1, 265)], nrmses, color='black')
    plt.ylim((0.0, 1.0))

    plt.xlabel('Edges made undirected')
    plt.ylabel('NARMA-10 NRMSE')

    plt.tight_layout()
    save_plot('undir-performance.png')
    plt.show()


def plot_making_edges_undirected():
    nrmses = pickle.load(open('experiments/changed_edges_nrmses.pkl', 'rb'))
    lattices = pickle.load(open('experiments/changed_edges_lattices.pkl', 'rb'))

    for i in [20, 100, 150, 225]:
        title = f'Lattice, {i} changed edges, NRMSE {nrmses[i]:.3f}'
        plot_lattice(lattices[i].reverse(), color_directed=True, title=title)


def plot_good_performance():
    lattices = pickle.load(open('experiments/grow_actual_lattices.pkl', 'rb'))
    ds.dataset = pickle.load(open('dataset/ds_narma_grow.pkl', 'rb'))
    esn = from_square_G(lattices[-1])
    plt.title(f'Prediction of a grown network with {esn.hidden_nodes} hidden nodes', y=-0.25)
    evaluate_esn(ds.dataset, esn, plot=True, plot_range=[0, 100])


def harder_benchmarks_performance():
    import dataset as ds

    u_train, y_train = ds.NARMA(sample_len = 2000, system_order=20)
    u_test, y_test = ds.NARMA(sample_len = 3000, system_order=20)
    dataset = [u_train, y_train, u_test, y_test]
    ds.dataset = dataset

    params = OrderedDict()
    params['w_res_type'] = ['tetragonal']
    params['hidden_nodes'] = [15*15]
    params['input_scaling'] = np.linspace(1/10000, 1, 150)
    params['w_in_distrib'] = [Distribution.fixed]
    params['dir_frac'] = [1.0]

    df = experiment(esn_nrmse, params, runs=10)
    df.to_pickle('experiments/narma20_performance.pkl')

    u_train, y_train = ds.NARMA(sample_len = 2000, system_order=30)
    u_test, y_test = ds.NARMA(sample_len = 3000, system_order=30)
    dataset = [u_train, y_train, u_test, y_test]
    ds.dataset = dataset

    params = OrderedDict()
    params['w_res_type'] = ['tetragonal']
    params['hidden_nodes'] = [15*15]
    params['input_scaling'] = np.linspace(1/10000, 1, 150)
    params['w_in_distrib'] = [Distribution.fixed]
    params['dir_frac'] = [1.0]

    df = experiment(esn_nrmse, params, runs=10)
    df.to_pickle('experiments/narma30_performance.pkl')


def plot_harder_benchmarks_performance():
    n20_df = load_experiment('experiments/narma20_performance.pkl')
    n20_df = n20_df.groupby(['input_scaling']).mean().reset_index()

    n30_df = load_experiment('experiments/narma30_performance.pkl')
    n30_df = n30_df.groupby(['input_scaling']).mean().reset_index()

    plt.xlabel('Input scaling')
    plt.ylabel('NRMSE')
    plt.title('NARMA performance vs. input scaling')

    plt.plot(n20_df['input_scaling'], n20_df['esn_nrmse'], label='NARMA20')
    plt.plot(n30_df['input_scaling'], n30_df['esn_nrmse'], label='NARMA30')

    plt.legend()
    plt.show()


def grow_shift_register_narma30():
    u_train, y_train = ds.NARMA(sample_len = 2000, system_order=30)
    u_test, y_test = ds.NARMA(sample_len = 3000, system_order=30)
    dataset = [u_train, y_train, u_test, y_test]
    ds.dataset = dataset

    lattices = []

    esn = create_delay_line(40)
    while True:
        node, edges = find_best_node_to_add(ds.dataset, esn)
        esn.add_hidden_node(node, edges)
        lattices.append(esn.G.copy())
        nrmse = evaluate_esn(ds.dataset, esn)

        print()
        print(f'nrmse after adding {len(lattices)} nodes ({esn.hidden_nodes} hidden): {nrmse}')
        pickle.dump(lattices, open('experiments/grow_dl_narma30.pkl', 'wb'))

        if esn.hidden_nodes >= 250:
            break


def plot_shift_register_lattices():
    # Lattices.
    lattices = pickle.load(open('experiments/grow_dl_narma30.pkl', 'rb'))

    set_figsize(10, 6)

    for i in [0, 1, 5, 15, -1]:
        lattice = lattices[i]
        esn = from_square_G(lattice)
        nrmse = evaluate_esn(ds.dataset, esn)
        title = f'{len(lattice.nodes)} hidden nodes, NRMSE: {nrmse}'
        plot_lattice(lattice, title=title)

    set_figsize(default_w, default_h)
