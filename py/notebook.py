import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import networkx as nx
from collections import OrderedDict

import dataset as ds
from ESN import ESN, Distribution, from_square_G
from metric import esn_nrmse, evaluate_esn
from gridsearch import experiment, load_experiment
from plot import plot_df_trisurf, set_figsize, get_figsize, plot_lattice, plot_vector_hist
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

    plot_lattice(dir_esn.G.reverse())


def plot_global_input_activations():
    set_figsize(14, 6)

    params = OrderedDict()
    params['hidden_nodes'] = 144
    params['w_res_density'] = 0.1
    def_esn = ESN(**params)

    dir_esn = pickle.load(open('models/dir_esn.pkl', 'rb'))
    dir_esn.w_in = torch.ones(dir_esn.hidden_nodes)
    dir_esn.w_in *= 0.1

    def_nrmse = evaluate_esn(ds.dataset, def_esn)
    dir_nrmse = evaluate_esn(ds.dataset, dir_esn)

    l = 30
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.suptitle('Activations for an ESN vs. square lattice')

    ax1.plot(def_esn.X[def_esn.washout:def_esn.washout+l])
    ax1.set_title('ESN')

    ax2.plot(dir_esn.X[dir_esn.washout:dir_esn.washout+l])
    ax2.set_title('Lattice')

    plt.show()

    set_figsize(default_w, default_h)


def node_removal_impact():
    dir_esn = pickle.load(open('models/dir_esn.pkl', 'rb'))
    dir_esn.set_readout('rr')
    nrmse_diffs = np.array(pickle.load(open('experiments/dir_diff_nrmses.pkl', 'rb')))

    def_nrmse = evaluate_esn(ds.dataset, dir_esn)
    max_clip = 1 - def_nrmse
    np.clip(nrmse_diffs, -1, max_clip, out=nrmse_diffs)

    title = 'Importance when removing single node (black = low importance)'
    plot_lattice(dir_esn.G, cols=nrmse_diffs, cmap_r=True, title=title)

    set_figsize(10, 6)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.suptitle('Distribution of absolut diff in NRMSE (right clips outliers)')
    plot_vector_hist(nrmse_diffs, n_bins=50, m=None, ax=ax1, show=False)
    plot_vector_hist(nrmse_diffs, n_bins=50, m=10.0, ax=ax2, show=True)
    set_figsize(default_w, default_h)


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


def remove_esn_nodes_performance():
    esn_nrmses = pickle.load(open('experiments/esn_removed_nodes_nrmses.pkl', 'rb'))
    lattice_nrmses = pickle.load(open('experiments/deg_lattice_nrmses.pkl', 'rb'))

    max_nodes = len(lattice_nrmses)

    print(f'Default min NRMSE: {max_nodes - np.argmin(esn_nrmses)} nodes with NRMSE {min(esn_nrmses)}')
    print(f'Lattice min NRMSE: {max_nodes - np.argmin(lattice_nrmses)} nodes with NRMSE {min(lattice_nrmses)}')

    x = list(range(len(esn_nrmses), 0, -1))
    plt.plot(x, esn_nrmses, label='ESN')
    plt.plot(x, lattice_nrmses, label='Lattice')

    plt.gca().invert_xaxis()
    plt.ylim((0.0, 1.0))

    plt.xlabel('Hidden nodes')
    plt.ylabel('NRMSE')

    plt.legend()
    plt.show()


def plot_node_removal():
    dir_esn = pickle.load(open('models/dir_esn.pkl', 'rb'))
    lattice_nrmses = pickle.load(open('experiments/deg_lattice_nrmses.pkl', 'rb'))
    lattices = pickle.load(open('experiments/deg_lattices.pkl', 'rb'))

    max_nodes = len(lattices)

    for i in [130, 70, 35, 20]:
        title = f'Lattice, {i} nodes, NRMSE {lattice_nrmses[max_nodes-i]:.3f}'
        plot_lattice(dir_esn.G.reverse(), alpha=0.5, show=False, ax=plt.gca())
        plot_lattice(lattices[max_nodes - i].reverse(), ax=plt.gca(), title=title)


def plot_esn_node_removal():
    esns = pickle.load(open('experiments/esn_removed_nodes_esns.pkl', 'rb'))
    esn_nrmses = pickle.load(open('experiments/esn_removed_nodes_nrmses.pkl', 'rb'))

    linear_esn = esns[len(esns)-31]
    G = nx.from_numpy_matrix(linear_esn.numpy())
    nx.draw(G, pos=nx.spring_layout(G))
    plt.show()


def plot_growth():
    # Lattices.
    lattices = pickle.load(open('experiments/grow_actual_lattices.pkl', 'rb'))
    ds.dataset = pickle.load(open('dataset/ds_narma_grow.pkl', 'rb'))

    set_figsize(10, 6)

    for i in [0, 50, -1]:
        lattice = lattices[i]
        esn = from_square_G(lattice)
        nrmse = evaluate_esn(ds.dataset, esn)
        title = f'{len(lattice.nodes)} hidden nodes, NRMSE: {nrmse}'
        plot_lattice(lattice, title=title)

    set_figsize(default_w, default_h)

    # NRMSEs.
    nrmses = pickle.load(open('experiments/grow_mid_nrmses.pkl', 'rb'))

    plt.title('Growing of original lattice of size 74')

    plt.xlabel('Nodes added')
    plt.ylabel('NRMSE')

    plt.plot(nrmses)
    plt.show()


def plot_making_edges_undirected_performance():
    nrmses = pickle.load(open('experiments/changed_edges_nrmses.pkl', 'rb'))

    edges = len(nrmses)
    print(f'To undirected min NRMSE: {np.argmin(nrmses)} edges removed with NRMSE {min(nrmses)}')

    plt.plot(nrmses)

    plt.ylim((0.0, 1.0))

    plt.xlabel('Edges made undirected')
    plt.ylabel('NRMSE')

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
