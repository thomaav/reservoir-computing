import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from collections import OrderedDict

import dataset as ds
from ESN import ESN, Distribution
from metric import esn_nrmse, evaluate_esn
from gridsearch import experiment, load_experiment
from plot import plot_df_trisurf, set_figsize, get_figsize, plot_lattice
from matrix import euclidean, inv, inv_squared, inv_cubed

default_w, default_h = get_figsize()


# GENERAL.


def esn_general_performance():
    params = OrderedDict()
    params['hidden_nodes'] = np.arange(20, 260, 10)
    params['w_res_density'] = [0.1]
    params['readout'] = ['rr']

    df = experiment(esn_nrmse, params, runs=20)
    df.to_pickle('experiments/esn_general_performance.pkl')
    pass


# EXPERIMENTS: Random Geometric Graphs.


def rgg_dist_performance():
    params = OrderedDict()
    params['w_res_type'] = ['waxman']
    params['hidden_nodes'] = np.arange(20, 90, 10)
    params['dist_function'] = [euclidean, inv, inv_squared, inv_cubed]
    params['readout'] = ['rr']

    df = experiment(esn_nrmse, params, runs=20)
    df.to_pickle('experiments/rgg_dist_performance.pkl')


def plot_rgg_dist_performance(agg='mean'):
    df = load_experiment('experiments/rgg_dist_performance.pkl')

    df['dist_function'] = df['dist_function'].apply(
        lambda f: f.__name__ if not isinstance(f, str) else f
    )

    euc_perf = df.loc[df['dist_function'] == euclidean.__name__]
    inv_perf = df.loc[df['dist_function'] == inv.__name__]
    inv_squared_perf = df.loc[df['dist_function'] == inv_squared.__name__]
    inv_cubed_perf = df.loc[df['dist_function'] == inv_cubed.__name__]

    labels = ['d', '1/d', '1/d^2', '1/d^3']

    for i, df in enumerate([euc_perf, inv_perf, inv_squared_perf, inv_cubed_perf]):
        if agg == 'mean':
            grouped_df = df.groupby(['hidden_nodes']).mean().reset_index()
        elif agg == 'min':
            grouped_df = df.groupby(['hidden_nodes']).min().reset_index()

        plt.plot(grouped_df['hidden_nodes'], grouped_df['esn_nrmse'], label=labels[i])

    plt.title(f'NRMSE for RGG with different distance functions, agg={agg}')
    plt.xlabel('Hidden nodes')
    plt.ylabel('NRMSE')

    plt.legend()
    plt.show()


# EXPERIMENTS: Regular Tilings.


def plot_regular_tilings():
    from ESN import ESN
    from plot import plot_lattice

    set_figsize(14, 6)

    esn_square = ESN(hidden_nodes=25, w_res_type='tetragonal')
    esn_hex = ESN(hidden_nodes=25, w_res_type='hexagonal')
    esn_tri = ESN(hidden_nodes=25, w_res_type='triangular')
    esn_rect = ESN(hidden_nodes=25, w_res_type='rectangular', rect_ratio=2.0)

    G_square = esn_square.G
    G_hex = esn_hex.G
    G_tri = esn_tri.G
    G_rect = esn_rect.G

    fig, axs = plt.subplots(2, 2)
    ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    plot_lattice(G_square, title='Square', ax=ax1, show=False)
    plot_lattice(G_hex, title='Hexagonal', ax=ax2, show=False)
    plot_lattice(G_tri, title='Triangular', ax=ax3, show=False)
    plot_lattice(G_rect, title='Rectangular', ax=ax4, show=True)

    set_figsize(default_w, default_h)


def regular_tilings_performance():
    pass


def plot_regular_tilings_performance():
    nrmse_df = load_experiment('experiments/lattice_nrmse.pkl')
    grouped_df = nrmse_df.groupby(['hidden_nodes', 'w_res_type']).mean().reset_index()

    tetragonal = grouped_df.loc[grouped_df['w_res_type'] == 'tetragonal']
    hexagonal = grouped_df.loc[grouped_df['w_res_type'] == 'hexagonal']
    triangular = grouped_df.loc[grouped_df['w_res_type'] == 'triangular']

    plt.plot(tetragonal['hidden_nodes'], tetragonal['esn_nrmse'], label='sq')
    plt.plot(hexagonal['hidden_nodes'], hexagonal['esn_nrmse'], label='hex')
    plt.plot(triangular['hidden_nodes'], triangular['esn_nrmse'], label='tri')

    plt.legend(fancybox=False, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.ylabel('NRMSE')
    plt.xlabel('Hidden nodes')

    plt.show()


def directed_regular_tilings_performance():
    pass


def plot_directed_regular_tilings_performance():
    lattice_dir_df = load_experiment('experiments/lattice_dir.pkl')
    esn_df = load_experiment('experiments/esn_general_performance.pkl')

    tetragonal = lattice_dir_df.loc[lattice_dir_df['w_res_type'] == 'tetragonal']
    hexagonal = lattice_dir_df.loc[lattice_dir_df['w_res_type'] == 'hexagonal']
    triangular = lattice_dir_df.loc[lattice_dir_df['w_res_type'] == 'triangular']

    groupby = ['hidden_nodes', 'dir_frac']
    axes    = ['hidden_nodes', 'dir_frac', 'esn_nrmse']
    agg     = ['mean', 'min']
    zlim    = (0.25, 0.7)
    azim    = 45
    titles  = ['tetragonal', 'hexagonal', 'triangular']

    for i, df in enumerate([tetragonal, hexagonal, triangular]):
        plot_df_trisurf(df=df, groupby=groupby, axes=axes, azim=azim, agg=agg,
                        zlim=zlim, title=titles[i])

    tetragonal = tetragonal.loc[tetragonal['dir_frac'] == 1.0]
    tetragonal = tetragonal.groupby(['hidden_nodes', 'dir_frac']).mean().reset_index()
    hexagonal = hexagonal.loc[hexagonal['dir_frac'] == 1.0]
    hexagonal = hexagonal.groupby(['hidden_nodes', 'dir_frac']).mean().reset_index()
    triangular = triangular.loc[triangular['dir_frac'] == 1.0]
    triangular = triangular.groupby(['hidden_nodes', 'dir_frac']).mean().reset_index()
    esn = esn_df.groupby(['hidden_nodes']).mean().reset_index()

    plt.plot(tetragonal['hidden_nodes'], tetragonal['esn_nrmse'], label='sq')
    plt.plot(hexagonal['hidden_nodes'], hexagonal['esn_nrmse'], label='hex')
    plt.plot(triangular['hidden_nodes'], triangular['esn_nrmse'], label='tri')
    plt.plot(esn['hidden_nodes'], esn['esn_nrmse'], label='esn')

    plt.legend(fancybox=False, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.title('NRMSE of regular tilings with all edges directed, agg=mean')
    plt.ylabel('NRMSE')
    plt.xlabel('Hidden nodes')

    plt.show()


def global_input_scheme_performance():
    params = OrderedDict()
    params['hidden_nodes'] = 144
    params['w_res_density'] = 0.1
    def_esn = ESN(**params)

    dir_esn = pickle.load(open('models/dir_esn.pkl', 'rb'))
    dir_esn.w_in = torch.ones(dir_esn.hidden_nodes)
    dir_esn.w_in *= 0.1

    def_nrmse = evaluate_esn(ds.dataset, def_esn)
    dir_nrmse = evaluate_esn(ds.dataset, dir_esn)

    print('NRMSE:')
    print(' default:', def_nrmse)
    print(' lattice:', dir_nrmse)

    print()
    print('Lattice weights:')
    print(f' w_in:  unique-{np.unique(dir_esn.w_in)}')
    print(f' w_res: unique-{np.unique(dir_esn.w_res)}')

    plot_lattice(dir_esn.G.reverse())


def plot_global_input_activations():
    l = 100
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.suptitle('Activations for an undirected vs. directed lattice')

    ax1.plot(undir_esn.X[undir_esn.washout:undir_esn.washout+l])
    ax1.set_title('Undirected')

    ax2.plot(dir_esn.X[dir_esn.washout:dir_esn.washout+l])
    ax2.set_title('Directed')

    plt.show()
