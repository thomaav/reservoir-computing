# %% [markdown]
"""
# Setup
"""

# %%
# Imports.
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pickle
import os
from tabulate import tabulate

# Important for multiprocessing.
import torch
torch.set_num_threads(1)

# General plotting things.
from plot import plot_lattice

# Experiment imports.
from gridsearch import load_experiment

# Dataset.
import dataset as ds

u_train, y_train = ds.NARMA(sample_len = 2000)
u_test, y_test = ds.NARMA(sample_len = 3000)
dataset = [u_train, y_train, u_test, y_test]
ds.dataset = dataset

# Oftentimes for debugging purposes.
from ESN import ESN, Distribution, from_square_G
from metric import evaluate_esn, memory_capacity

# from notebook import default_plot_settings
# default_plot_settings()
plt.style.use('./paper.mplstyle')


textwidth = 7.02625 # inches
columnwidth = 3.32492 # inches
figsize_full = (textwidth, textwidth * 0.75)

FIG_DIR = '../paper/figures/'
def savefig(filename, **kwargs):
    #plt.subplots_adjust(left=0.1, right=0.9)
    #plt.savefig(os.path.join(figpath, filename), pad_inches=0)
    #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
    #            hspace = 0, wspace = 0)

    fig = plt.gcf()
    width, height = fig.get_size_inches()*72
    print(f"Saving {filename}: {width:.1f}x{height:.1f}")
    plt.savefig(os.path.join(FIG_DIR, filename), **kwargs)#, bbox_inches='tight', pad_inches=0)

# %% [markdown]
"""
# Experiments: Regular Tilings
"""

# %%
from notebook import paper_esn_general_performance
paper_esn_general_performance()

# %%
# Compare old (input_scaling=1) with new (input_scaling=0.1)
esn = load_experiment('experiments/esn_general_performance.pkl')
esn.drop(columns=['readout'], inplace=True)
esn = esn.groupby(['hidden_nodes']).mean().reset_index()
plt.plot(esn['hidden_nodes'], esn['esn_nrmse'], label='ESN')

esn = load_experiment('experiments/paper_esn_general_performance.pkl')
esn.drop(columns=['readout'], inplace=True)
esn = esn.groupby(['hidden_nodes']).mean().reset_index()
plt.plot(esn['hidden_nodes'], esn['esn_nrmse'], label='ESN 0.1')

plt.legend()

# %% [markdown]
"""
## Lattice topologies
"""

# %%
import matrix

np.random.seed(0)

esn_square = ESN(hidden_nodes=5**2, w_res_type='tetragonal', dir_frac=1)
esn_hex = ESN(hidden_nodes=4**2, w_res_type='hexagonal', dir_frac=1)
esn_tri = ESN(hidden_nodes=3**2, w_res_type='triangular', dir_frac=1)

print(esn_square.hidden_nodes, esn_hex.hidden_nodes, esn_tri.hidden_nodes)

G_square = esn_square.G
G_hex = esn_hex.G
G_tri = esn_tri.G

# Make graphs manually for better presentable sizes
G_square = matrix.tetragonal([5, 5])
G_square = matrix.make_graph_directed(G_square, 1.0)
G_hex = matrix.hexagonal(3, 3)
G_hex = matrix.make_graph_directed(G_hex, 1.0)
G_tri = matrix.triangular(5, 6)
#G_tri = matrix.make_graph_directed(G_tri, 1.0)

print(len(G_square), len(G_hex), len(G_tri))

sz = textwidth
figsize = (sz/3, sz/3)

file_names = ['square', 'hex', 'triangular']
for i, G in enumerate([G_square, G_hex, G_tri]):
    fig, ax = plt.subplots(figsize=figsize)
    plot_lattice(G, hide_axes=True, color_directed=False, show=False, ax=ax)

    savefig(file_names[i] + '.svg', bbox_inches='tight', pad_inches=0)
    savefig(file_names[i] + '.pdf', bbox_inches='tight', pad_inches=0)
    plt.show()

# %% [markdown]
"""
## Undirected lattice reservoirs
"""

# %%
from notebook import regular_tilings_performance_is01
regular_tilings_performance_is01()

# %%
df = load_experiment('experiments/lattice_nrmse_is0.1.pkl')
esn = load_experiment('experiments/paper_esn_general_performance.pkl')
esn = esn[esn['hidden_nodes'] <= df['hidden_nodes'].max()]

assert len(df) == 3*20*len(df['hidden_nodes'].unique())
assert len(esn) == 20*len(esn['hidden_nodes'].unique())
assert set(df['hidden_nodes']) == set(esn['hidden_nodes'])

sq = df[df['w_res_type'] == 'tetragonal']
hex = df[df['w_res_type'] == 'hexagonal']
tri = df[df['w_res_type'] == 'triangular']

# Scatter plot
plt.figure()
plt.scatter(sq['hidden_nodes'], sq['esn_nrmse'], label='Square')
plt.scatter(hex['hidden_nodes'], hex['esn_nrmse'], label='Hex')
plt.scatter(tri['hidden_nodes'], tri['esn_nrmse'], label='Tri')
plt.scatter(esn['hidden_nodes'], esn['esn_nrmse'], label='ESN')
plt.legend()

def plot_nrmse(df, label, x='hidden_nodes', plot_std=False, verbose=True, **kwargs):
    mean  =  df.groupby([x]).mean(numeric_only=True).esn_nrmse
    std  =  df.groupby([x]).std(numeric_only=True).esn_nrmse

    if verbose:
        print(f"{label} NRMSE mean {mean.mean():.4f} (pooled std: {np.sqrt(np.mean(std**2)):.4f})")
        xmax=df[x].max()
        print(f"{label} {x}={xmax} NRMSE mean {mean.loc[xmax]:.4f} (std: {std.loc[xmax]:.4f})")

    plt.plot(mean.index, mean, label=label, **kwargs)

    if plot_std:
        plt.fill_between(mean.index, mean-std, mean+std, color='black', alpha=0.2, lw=0)

# With std (too busy)
sz = .5*textwidth
figsize = (sz, sz/1.618)
plt.figure(figsize=figsize)

plot_nrmse(sq, 'Square', plot_std=True, color='black', linestyle='solid')
plot_nrmse(hex, 'Hexagonal', plot_std=True, color='black', linestyle='dashed')
plot_nrmse(tri, 'Triangular', plot_std=True, color='black', linestyle='dotted')
plot_nrmse(esn, 'ESN', plot_std=True, color='black', linestyle='dashdot')

plt.legend(fancybox=False, loc='upper right', bbox_to_anchor=(1.0, 1.0))
plt.ylabel('NRMSE')
plt.xlabel('Reservoir size $N$')

# Without std
sz = columnwidth
figsize = (sz, sz/1.618)
plt.figure(figsize=figsize, layout='constrained')

plot_nrmse(sq, 'Square', color='black', linestyle='solid')
plot_nrmse(hex, 'Hexagonal', color='black', linestyle='dashed')
plot_nrmse(tri, 'Triangular', color='black', linestyle='dotted')
plot_nrmse(esn, 'ESN', color='black', linestyle='dashdot')

plt.legend(fancybox=False, loc='upper right', bbox_to_anchor=(1.0, 1.0))
plt.ylabel('NRMSE')
plt.xlabel('Reservoir size $N$')

savefig('lattice_nrmse_is0.1.svg')
savefig('lattice_nrmse_is0.1.pdf')

# %%
# Plot old results (input_scaling=1):
from notebook import plot_regular_tilings_performance
plot_regular_tilings_performance(std=False)

# %%
# Rerun with runs=20
from notebook import regular_tilings_performance_is
regular_tilings_performance_is()

# %%
from notebook import plot_regular_tilings_performance_is
plot_regular_tilings_performance_is(std=False)

# %%
# Plot one line per input scaling
df = load_experiment("experiments/rt_performance_is.pkl")
df = df[df['input_scaling'] <= 1]

assert len(df) == 3*20*len(df['hidden_nodes'].unique())*len(df['input_scaling'].unique())

for grp, dfi in df.groupby('w_res_type'):
    plt.figure()
    plt.title(grp)
    for input_scaling, dfj in dfi.groupby('input_scaling'):
        plot_nrmse(dfj, label=f"is={input_scaling:.2f}", verbose=False)
    plt.legend()

plt.figure()
df = df[df['input_scaling'] == .1]
for grp, dfi in df.groupby('w_res_type'):
    plot_nrmse(dfi, label=grp)

# esn = load_experiment('experiments/esn_general_performance.pkl')
# esn = esn[esn['hidden_nodes'] <= df['hidden_nodes'].max()]
# esn = esn.groupby(['hidden_nodes']).mean(numeric_only=True).reset_index()
# plt.plot(esn['hidden_nodes'], esn['esn_nrmse'], label='ESN')

esn = load_experiment('experiments/paper_esn_general_performance.pkl')
esn = esn[esn['hidden_nodes'] <= df['hidden_nodes'].max()]
plot_nrmse(esn, label='ESN 0.1')

plt.legend()

# %% [markdown]
"""
# Where does diversity come from?

Two potential sources:
1. Non-global input
2. Edge effects
"""

# %%
from notebook import regular_tilings_performance_is01_global
regular_tilings_performance_is01_global()

# %%
from notebook import regular_tilings_performance_is01_periodic
regular_tilings_performance_is01_periodic()

# %%
from notebook import regular_tilings_performance_is01_global_periodic
regular_tilings_performance_is01_global_periodic()

# %%
from notebook import regular_tilings_performance_is001
regular_tilings_performance_is001()

# %%
from notebook import regular_tilings_performance_is001_global
regular_tilings_performance_is001_global()

# %%
from notebook import regular_tilings_performance_is001_periodic
regular_tilings_performance_is001_periodic()

# %%
from notebook import regular_tilings_performance_is001_global_periodic
regular_tilings_performance_is001_global_periodic()

# %%
def do_plot(filename):
    df = load_experiment(filename)
    esn = load_experiment('experiments/paper_esn_general_performance.pkl')
    esn = esn[esn['hidden_nodes'] <= df['hidden_nodes'].max()]

    # assert len(df) == 3*len(df['hidden_nodes'].unique())
    assert len(esn) == 20*len(esn['hidden_nodes'].unique())
    # assert set(df['hidden_nodes']) == set(esn['hidden_nodes'])

    sq = df[df['w_res_type'] == 'tetragonal']
    hex = df[df['w_res_type'] == 'hexagonal']
    tri = df[df['w_res_type'] == 'triangular']

    sz = .7*textwidth
    figsize = (sz, sz/1.618)
    plt.figure(figsize=figsize)

    plot_nrmse(sq, label='Square', color='black', linestyle='solid')
    plot_nrmse(hex, label='Hexagonal', color='black', linestyle='dashed')
    plot_nrmse(tri, label='Triangular', color='black', linestyle='dotted')
    plot_nrmse(esn, label='ESN', color='black', linestyle='dashdot')

    plt.legend(fancybox=False, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.ylabel('NRMSE')
    plt.xlabel('Reservoir size $N$')
    #plt.ylim(0.1, 0.8)

print('Local input (0.1)')
do_plot('experiments/lattice_nrmse_is0.1.pkl')
plt.title('Local input (0.1)')

print('\nGlobal input (0.1)')
do_plot('experiments/lattice_nrmse_is0.1_global.pkl')
plt.title('Global input (0.1)')

print('\nPeriodic (0.1)')
do_plot('experiments/lattice_nrmse_is0.1_periodic.pkl')
plt.title('Periodic (0.1)');

print('\nGlobal + periodic (0.1)')
do_plot('experiments/lattice_nrmse_is0.1_global_periodic.pkl')
plt.title('Global + periodic (0.1)');

print('\nLocal input (0.01)')
do_plot('experiments/lattice_nrmse_is0.01.pkl')
plt.title('Local input (0.01)')

print('\nGlobal input (0.01)')
do_plot('experiments/lattice_nrmse_is0.01_global.pkl')
plt.title('Global input (0.01)')

print('\nPeriodic (0.01)')
do_plot('experiments/lattice_nrmse_is0.01_periodic.pkl')
plt.title('Periodic (0.01)');

print('\nGlobal + periodic (0.01)')
do_plot('experiments/lattice_nrmse_is0.01_global_periodic.pkl')
plt.title('Global + periodic (0.01)');

# %% [markdown]
"""
Random **input weights** is the primary source of diversity in undirected lattice networks.
However, reducing input scaling to 0.01 helps somewhat.
But in both cases, global input is detrimental to performance.
Periodic boundaries, on the other hand, seems to have little effect.

## What about memory capacity?
"""

# %%
from ESN import ESN, Distribution
from metric import evaluate_esn, memory_capacity

params = {
    'undirected': dict(
        hidden_nodes=20**2,
        w_res_type='tetragonal',
        input_scaling=0.1,
    ),

    'undirected global': dict(
        hidden_nodes=20**2,
        w_res_type='tetragonal',
        input_scaling=0.1,
        w_in_distrib=Distribution.fixed,
    ),

    'undirected periodic': dict(
        hidden_nodes=20**2,
        w_res_type='tetragonal',
        input_scaling=0.1,
        periodic=True,
    ),

    'undirected global periodic': dict(
        hidden_nodes=20**2,
        w_res_type='tetragonal',
        input_scaling=0.1,
        periodic=True,
        w_in_distrib=Distribution.fixed,
    ),

    'ESN': dict(
        hidden_nodes=20**2,
        w_res_density=0.1,
        input_scaling=0.1,
    ),

    'ESN global': dict(
        hidden_nodes=20**2,
        w_res_density=0.1,
        input_scaling=0.1,
        w_in_distrib=Distribution.fixed,
    ),
}

for k, kwargs in params.items():
    esn = ESN(**kwargs)

    # plt.figure()
    # plt.title(k)
    # plt.xlim(2000, 2050)

    score = evaluate_esn(dataset, esn, plot=False, show=False)
    print(f'NARMA10 NRMSE ({k}): {score}')

# %%
for k, kwargs in params.items():
    esn = ESN(w_ridge=0.001, **kwargs)
    print(f'MC ({k}):', memory_capacity(esn))
    plt.plot(esn.mcs, '.-', label=k)

plt.legend()
plt.xlim(0, 20);

# %%
from plot import plot_lattice
plt.figure(figsize=(6, 4))
# Turn up w_ridge to force some structure to appear
k = 'undirected global'
esn = ESN(w_ridge=0.001, **params[k])
print(f'MC ({k}):', memory_capacity(esn))
#plt.plot(esn.mcs, '.-', label=k)
for k in range(6):
    plt.subplot(2, 3, k+1)
    cols = esn.w_outs[k].coef_
    #cols = np.log(np.abs(cols))
    vlim = np.max(np.abs(cols))
    print(f"k={k} vlim={vlim}")
    plot_lattice(esn.G, hide_axes=True, show=False, cols=cols, cmap='RdBu', vmin=-vlim, vmax=vlim, ax=plt.gca())
    plt.title(f"k={k}")

# %%
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

fig, ax = plt.subplots(figsize=(6, 4))

def animate(i):
    ax.cla()
    cols = esn.X[esn.washout+i].numpy()
    #cols = np.log(np.abs(cols))
    # plot_lattice(esn.G, hide_axes=True, show=False, cols=cols, cmap='viridis', ax=ax)
    vlim = np.max(np.abs(cols))
    plot_lattice(esn.G, hide_axes=True, show=False, cols=cols, cmap='RdBu', vmin=-vlim, vmax=vlim, ax=ax)
    ax.set_title(f"{i}, {np.min(cols)}, {np.max(cols)}")

anim = FuncAnimation(fig, animate, frames=100, interval=100, blit=False)
plt.close()
#plt.show()
HTML(anim.to_jshtml())

# %% [markdown]
"""
# Regular tilings with periodic boundaries
"""

# %%
from notebook import regular_tilings_performance_is_periodic
regular_tilings_performance_is_periodic()

# %%
from notebook import plot_regular_tilings_performance_is
plot_regular_tilings_performance_is(
    std=False,
    filename='experiments/rt_performance_is_periodic.pkl',
    save='regular-tilings-performance-is-periodic')

# %% [markdown]
"""
# Regular tilings with global input
"""

# %%
from notebook import regular_tilings_performance_is_global
regular_tilings_performance_is_global()

# %%
from notebook import plot_regular_tilings_performance_is
plot_regular_tilings_performance_is(
    std=False,
    filename='experiments/rt_performance_is_global.pkl',
    save='regular-tilings-performance-is-global')

# %% [markdown]
"""
# Regular tilings with periodic boundaries and global input
"""

# %%
from notebook import regular_tilings_performance_is_periodic_global
regular_tilings_performance_is_periodic_global()

# %%
from notebook import plot_regular_tilings_performance_is
plot_regular_tilings_performance_is(
    std=False,
    filename='experiments/rt_performance_is_periodic_global.pkl',
    save='regular-tilings-performance-is-periodic-global')

# %% [markdown]
"""
## Directed lattice reservoirs
"""

# %%
from notebook import plot_directed_lattice
plot_directed_lattice()

# %%
from notebook import directed_regular_tilings_performance
directed_regular_tilings_performance()

# %%
from notebook import plot_directed_regular_tilings_performance
plot_directed_regular_tilings_performance(std=False)

# %%
from notebook import directed_regular_tilings_performance_dir_frac
directed_regular_tilings_performance_dir_frac()

# %%
from notebook import directed_regular_tilings_performance_dir_frac_global
directed_regular_tilings_performance_dir_frac_global()

# %%
from notebook import directed_regular_tilings_performance_dir_full
directed_regular_tilings_performance_dir_full()

# %%
df = load_experiment('experiments/lattice_dir_frac.pkl')

assert len(df) == 3*20*len(df['dir_frac'].unique())

# Plot: dir_frac vs performance
sq = df[df['w_res_type'] == 'tetragonal']
hex = df[df['w_res_type'] == 'hexagonal']
tri = df[df['w_res_type'] == 'triangular']

# sq = sq.groupby(['dir_frac']).mean(numeric_only=True)
# hex = hex.groupby(['dir_frac']).mean(numeric_only=True)
# tri = tri.groupby(['dir_frac']).mean(numeric_only=True)
#
plt.figure()
sq.plot(kind='scatter', x='dir_frac', y='esn_nrmse')
hex.plot(kind='scatter', x='dir_frac', y='esn_nrmse')
tri.plot(kind='scatter', x='dir_frac', y='esn_nrmse')

sz = textwidth/2
figsize = (sz, sz/1.618)
plt.figure(figsize=figsize, layout='constrained')

plot_nrmse(sq, 'Square', x='dir_frac', plot_std=False, color='black', linestyle='solid')
plot_nrmse(hex, 'Hexagonal', x='dir_frac', plot_std=False, color='black', linestyle='dashed')
plot_nrmse(tri, 'Triangular', x='dir_frac', plot_std=False, color='black', linestyle='dotted')

#plt.legend(fancybox=False, loc='upper right', bbox_to_anchor=(1.0, 1.0))
plt.legend(fancybox=False)
plt.ylabel('NRMSE')
plt.xlabel('Fraction of directed edges')
# plt.ylim(0, 1)

savefig('rt-dir-frac.svg')
savefig('rt-dir-frac.pdf')


# Plot: size vs performance for fully directed reservoirs
df = load_experiment('experiments/lattice_dir_full.pkl')
esn = load_experiment('experiments/paper_esn_general_performance.pkl')

assert len(df) == 3*20*len(df['hidden_nodes'].unique())
assert set(df['hidden_nodes']) == set(esn['hidden_nodes'])

sq = df[df['w_res_type'] == 'tetragonal']
hex = df[df['w_res_type'] == 'hexagonal']
tri = df[df['w_res_type'] == 'triangular']

plt.figure(figsize=figsize, layout='constrained')

plot_nrmse(sq, 'Square', color='black', linestyle='solid')
plot_nrmse(hex, 'Hexagonal', color='black', linestyle='dashed')
plot_nrmse(tri, 'Triangular', color='black', linestyle='dotted')
plot_nrmse(esn, 'ESN', color='black', linestyle='dashdot')

#plt.legend(fancybox=False, loc='upper right', bbox_to_anchor=(1.0, 1.0))
plt.legend(fancybox=False)
plt.ylabel('NRMSE')
plt.xlabel('Reservoir size $N$')

savefig('rt-dir-full.svg')
savefig('rt-dir-full.pdf')

# %%
esn = ESN(hidden_nodes=81, w_res_type='tetragonal')
evaluate_esn(ds.dataset, esn, plot=True, plot_range=[0, 100])

# %% [markdown]
"""
## Global input scheme
"""

# %%
from notebook import paper_directed_lattice_performance
paper_directed_lattice_performance()

# %%
from notebook import esn_global_input_performance
esn_global_input_performance()

# %%
sq_local = load_experiment('experiments/lattice_dir_full.pkl')
esn_local = load_experiment('experiments/paper_esn_general_performance.pkl')
sq_global = load_experiment('experiments/paper_directed_lattice_performance.pkl')
esn_global = load_experiment('experiments/esn_global_input_performance.pkl')

usq_local = load_experiment('experiments/lattice_nrmse_is0.01.pkl')
usq_global = load_experiment('experiments/lattice_nrmse_is0.01_global.pkl')

sq_local = sq_local[sq_local['w_res_type'] == 'tetragonal']
usq_local = usq_local[usq_local['w_res_type'] == 'tetragonal']
usq_global = usq_global[usq_global['w_res_type'] == 'tetragonal']

assert len(sq_local) == len(esn_local) == len(sq_global) == len(esn_global) == 20*len(sq_local['hidden_nodes'].unique())
assert set(sq_local['hidden_nodes']) == set(esn_local['hidden_nodes']) == set(sq_global['hidden_nodes']) == set(esn_global['hidden_nodes'])

sz = columnwidth
figsize = (sz, sz/1.618)

plt.figure(figsize=figsize, layout='constrained')

plot_nrmse(sq_local, "Square (local)", color='black', linestyle='solid')
plot_nrmse(sq_global, "Square (global)", color='black', linestyle='dashed')
plot_nrmse(esn_local, 'ESN', color='black', linestyle='dashdot')

plt.legend(fancybox=False, loc='upper right', bbox_to_anchor=(1.0, 1.0))
plt.ylabel('NRMSE')
plt.xlabel('Reservoir size $N$')
plt.ylim(0.1, 0.8)

savefig('square_global_nrmse.svg')
savefig('square_global_nrmse.pdf')


plt.figure(figsize=figsize, layout='constrained')

plot_nrmse(esn_local, "ESN (local)", color='black', linestyle='solid')
plot_nrmse(esn_global, "ESN (global)", color='black', linestyle='dashed')

plt.legend(fancybox=False, loc='upper right', bbox_to_anchor=(1.0, 1.0))
plt.ylabel('NRMSE')
plt.xlabel('Reservoir size $N$')
plt.ylim(0.1, 0.8)

savefig('esn_global_nrmse.svg')
savefig('esn_global_nrmse.pdf')


plt.figure(figsize=figsize, layout='constrained')

plot_nrmse(usq_local, "Undirected square (0.01 local)",   color='black', linestyle='solid')
plot_nrmse(usq_global, "Undirected square (0.01 global)", color='black', linestyle='dashed')

plt.legend(fancybox=False, loc='upper right', bbox_to_anchor=(1.0, 1.0))
plt.ylabel('NRMSE')
plt.xlabel('Reservoir size $N$')
plt.ylim(0.1, 0.8)

savefig('undirected_square_global_nrmse.svg')
savefig('undirected_square_global_nrmse.pdf')

# %%
from notebook import paper_unique_weights
paper_unique_weights()

# %%
from notebook import paper_print_unique_weights
paper_print_unique_weights()

# %%
from notebook import unique_weights
unique_weights()

# %%
from notebook import print_unique_weights
print_unique_weights()

# %% [markdown]
"""
## Reservoir robustness: removing nodes gradually
"""

# %%
# %%script false --no-raise-error

from notebook import node_removal_impact
node_removal_impact()

# %%
from notebook import plot_node_removal_impact
plot_node_removal_impact()

# %%
from notebook import remove_nodes_performance
remove_nodes_performance()

# %%
from notebook import paper_general_esn_shrink
paper_general_esn_shrink()

# %%
# From esn.ipynb
# Rerun with input_scaling=0.1
# This takes a long time...
from ESN import find_esn
from experiment import remove_nodes_incrementally

try:
    cur_esn = pickle.load(open('models/paper_esn_model_removed_nodes.pkl', 'rb'))
except FileNotFoundError:
    params = { 'hidden_nodes': 144, 'w_res_density': 0.10, 'input_scaling': 0.1 }
    cur_esn = find_esn(dataset=ds.dataset, required_nrmse=0.28, **params)
    pickle.dump(cur_esn, open('models/paper_esn_model_removed_nodes.pkl', 'wb'))

remove_nodes_incrementally(ds.dataset, cur_esn, 'experiments/paper_esn_removed_nodes.pkl')

# %%
# From esn.ipynb
# TODO: Why is this done in two steps?
from experiment import evaluate_incremental_node_removal

esn_file = 'models/paper_esn_model_removed_nodes.pkl'
removed_nodes_file = 'experiments/paper_esn_removed_nodes.pkl'

nrmses, esns = evaluate_incremental_node_removal(ds.dataset, esn_file, removed_nodes_file, return_esns=True)
esns = [esn.w_res for esn in esns]

pickle.dump(nrmses, open('experiments/paper_esn_removed_nodes_nrmses.pkl', 'wb'))
pickle.dump(esns, open('experiments/paper_esn_removed_nodes_esns.pkl', 'wb'))

# %%
esn_file = 'models/paper_esn_model_removed_nodes.pkl'
esn = pickle.load(open(esn_file, 'rb'))
print(esn.hidden_nodes, 12**2, esn.input_scaling)

esn_file = 'models/esn_model_removed_nodes.pkl'
import sklearn.linear_model
sys.modules['sklearn.linear_model.ridge'] = sklearn.linear_model
esn = pickle.load(open(esn_file, 'rb'))
print(esn.hidden_nodes, 12**2, esn.input_scaling)

esn_file = 'models/dir_esn.pkl'
esn = pickle.load(open(esn_file, 'rb'))
print(esn.hidden_nodes, 12**2, esn.input_scaling)

# %%
esn_shrink_nrmses = pickle.load(open('experiments/paper_esn_removed_nodes_nrmses.pkl', 'rb'))
lattice_shrink_nrmses = pickle.load(open('experiments/deg_lattice_nrmses.pkl', 'rb'))

esn_general = load_experiment('experiments/paper_esn_general_short.pkl')
esn_general = esn_general.groupby(['hidden_nodes']).mean().reset_index()
print(set(esn_general['hidden_nodes']))

max_nodes = len(lattice_shrink_nrmses)

sz = columnwidth
figsize = (sz, sz/1.618)
plt.figure(figsize=figsize, layout='constrained')

x = list(range(len(esn_shrink_nrmses), 0, -1))
print(set(x))
plt.plot(x, lattice_shrink_nrmses, label='Square', color='black', linestyle='solid')
plt.plot(x, esn_shrink_nrmses, label='ESN (shrinking)', color='black', linestyle='dotted')
plt.plot(esn_general['hidden_nodes'], esn_general['esn_nrmse'], label='Random ESN', color='black', linestyle='dashed')

plt.gca().invert_xaxis()
plt.ylim((0.0, 1.0))

plt.xlabel('Reservoir size $N$')
plt.ylabel('NRMSE')

plt.legend()

savefig('shrink-performance.svg')
savefig('shrink-performance.pdf')

# %%
# Old plot:
from notebook import remove_esn_nodes_performance
remove_esn_nodes_performance()


# %% [markdown]
"""
Now, how do these lattices look?
"""

# %%
dir_esn = pickle.load(open('models/dir_esn.pkl', 'rb'))
lattice_nrmses = pickle.load(open('experiments/deg_lattice_nrmses.pkl', 'rb'))
lattices = pickle.load(open('experiments/deg_lattices.pkl', 'rb'))

max_nodes = len(lattices)

sz = textwidth/2
figsize = (sz, sz)
for i in [130, 70, 35, 20]:
    title = f'Lattice, {i} nodes, NRMSE {lattice_nrmses[max_nodes-i]:.3f}'
    print(title)
    plt.figure(figsize=figsize)
    ax = plt.gca()
    plot_lattice(dir_esn.G.reverse(), edge_color='0.7', cols='0.7', show=False, ax=ax)
    plot_lattice(lattices[max_nodes - i].reverse(), show=False, ax=ax)
    x1, x2, y1, y2 = ax.axis()
    ax.axis((x1+1.5, x2-1.5, y1+1.5, y2-1.5))
    ax.set_axis_off()
    savefig(f'sq-grid-{i}.pdf', bbox_inches='tight', pad_inches=0.01)
    savefig(f'sq-grid-{i}.svg', bbox_inches='tight', pad_inches=0.01)

# %%
# Old plots
from notebook import plot_node_removal
plot_node_removal(save=False)

# %% [markdown]
"""
How does the ESN look?
"""

# %%
from notebook import plot_esn_node_removal
plot_esn_node_removal()

# %% [markdown]
"""
## Growing reservoirs: adding nodes incrementally

This is actually a new experiment, where
1. Find dir lattice reservoirESN with N=144 and NRMSE < 0.25
2. Shrink reservoir and evaluate performance
3. Choose reservoir at N=74, which is then grown to N=250
"""

# %%
esn_shrink_nrmses = pickle.load(open('experiments/paper_esn_removed_nodes_nrmses.pkl', 'rb'))
lattice_shrink_nrmses = pickle.load(open('experiments/deg_lattice_nrmses.pkl', 'rb'))
lattice_shrink_nrmses2 = pickle.load(open('experiments/grow_nrmses.pkl', 'rb'))

esn_general = load_experiment('experiments/paper_esn_general_short.pkl')
esn_general = esn_general.groupby(['hidden_nodes']).mean().reset_index()
print(set(esn_general['hidden_nodes']))

max_nodes = len(lattice_shrink_nrmses)

sz = textwidth/2
figsize = (sz, sz/1.618)
# plt.figure(figsize=figsize)

x = list(range(len(esn_shrink_nrmses), 0, -1))
print(set(x))
ibest = np.argmin(lattice_shrink_nrmses)
print(x[ibest], lattice_shrink_nrmses[ibest])
ibest = np.argmin(lattice_shrink_nrmses2)
print(x[ibest], lattice_shrink_nrmses2[ibest])
plt.plot(esn_general['hidden_nodes'], esn_general['esn_nrmse'], label='Random ESN', color='black', linestyle='dashed')
plt.plot(x, esn_shrink_nrmses, label='ESN (shrinking)', color='black', linestyle='dotted')
plt.plot(x, lattice_shrink_nrmses, label='Square', color='black', linestyle='solid')
plt.plot(x, lattice_shrink_nrmses2, label='Square (from growth)', color='green', linestyle='solid')

plt.gca().invert_xaxis()
plt.ylim((0.0, 1.0))

plt.xlabel('Hidden nodes remaining')
plt.ylabel('NARMA-10 NRMSE')

plt.legend()

# savefig('shrink-performance.svg', bbox_inches='tight', pad_inches=0.01)
# savefig('shrink-performance.pdf', bbox_inches='tight', pad_inches=0.01)
#
# %% [markdown]
"""
The graphs from the two shrinking experiments look pretty similar. What about the resulting networks?
"""

# %%
dir_esn = pickle.load(open('models/dir_lattice_grow.pkl', 'rb'))
lattice_nrmses = pickle.load(open('experiments/grow_nrmses.pkl', 'rb'))
lattices = pickle.load(open('experiments/grow_lattices.pkl', 'rb'))

max_nodes = len(lattices)

print(np.array(list(zip(range(max_nodes, 0, -1), lattice_nrmses))))

sz = textwidth/2
figsize = (sz, sz)
#for i in [130, 70, 35, 20]:
for i in [130, 70, 30, 20]:
    title = f'Lattice, {i} nodes, NRMSE {lattice_nrmses[max_nodes-i]:.3f}'
    print(title)
    plt.figure(figsize=figsize)
    ax = plt.gca()
    plot_lattice(dir_esn.G.reverse(), edge_color='0.7', cols='0.7', show=False, ax=ax)
    plot_lattice(lattices[max_nodes - i].reverse(), show=False, ax=ax)
    x1, x2, y1, y2 = ax.axis()
    ax.axis((x1+1.5, x2-1.5, y1+1.5, y2-1.5))
    ax.set_axis_off()
    #savefig(f'sq-grid-{i}.pdf', bbox_inches='tight', pad_inches=0.01)
    #savefig(f'sq-grid-{i}.svg', bbox_inches='tight', pad_inches=0.01)

# %% [markdown]
"""
They look good, but not as "shift register" like the previous results...

Two options:
1. Use results as is, but need to explain that growth starts from a new shrinking process.
2. Use growth shrinking data for both shrink and growth, but then the resulting networks aren't as nice.
"""

# %%
# Lattices.
lattices = pickle.load(open('experiments/grow_actual_lattices.pkl', 'rb'))
nrmses = pickle.load(open('experiments/grow_mid_nrmses.pkl', 'rb'))
ds.dataset = pickle.load(open('dataset/ds_narma_grow.pkl', 'rb'))

sz = textwidth*.7
figsize = (sz, sz)

for i in [0, 50, -1]:
    lattice = lattices[i]
    nrmse = nrmses[i]
    N = len(lattice.nodes)
    print(f'{N} hidden nodes, NRMSE: {nrmse}')
    pos = list(nx.get_node_attributes(lattice, 'pos').values())
    xmin, ymin = np.min(pos, axis=0)
    xmax, ymax = np.max(pos, axis=0)
    width = xmax - xmin
    height = ymax - ymin
    print(xmin, xmax, ymin, ymax)
    cx = xmin + width/2
    cy = ymin + height/2
    # sz = max(width, height)*.25
    print(sz)
    figsize = (sz, sz)
    plt.figure(figsize=figsize)
    ax = plt.gca()
    plot_lattice(lattice.reverse(), show=False, ax=ax)
    x1, x2, y1, y2 = ax.axis()
    print(x1, x2, y1, y2)
    # ax.axis((x1+1.5, x2-1.5, y1+1.5, y2-1.5))
    print((x1+1.5, x2-1.5, y1+1.5, y2-1.5))
    ax.axis((cx-11, cx+11, cy-11, cy+11))
    ax.set_axis_off()
    plt.tight_layout()
    savefig(f'sq-grid-grow-{N}.pdf', bbox_inches='tight', pad_inches=0.01)
    savefig(f'sq-grid-grow-{N}.svg', bbox_inches='tight', pad_inches=0.01)

# %%
from notebook import plot_growth
plot_growth(save=False)

# %%
nrmses = pickle.load(open('experiments/grow_mid_nrmses.pkl', 'rb'))

sz = columnwidth
figsize = (sz, sz/1.618)
plt.figure(figsize=figsize, layout='constrained')

plt.xlabel('Reservoir size $N$')
plt.ylabel('NRMSE')

plt.plot(range(74, 74+len(nrmses)), nrmses, color='black')
savefig(f'grow-performance.pdf')
savefig(f'grow-performance.svg')
plt.show()

# %% [markdown]
"""
## Gradually making edges undirected
"""

# %%
# This takes a long time...
from notebook import making_edges_undirected_performance
making_edges_undirected_performance()

# %%
nrmses = pickle.load(open('experiments/changed_edges_nrmses.pkl', 'rb'))

edges = len(nrmses)
print(f'To undirected min NRMSE: {np.argmin(nrmses)} edges removed with NRMSE {min(nrmses)}')

sz = columnwidth
figsize = (sz, sz/1.618)
plt.figure(figsize=figsize, layout='constrained')

plt.plot([n/264 for n in range(1, 265)], nrmses, color='black')
plt.ylim((0.0, 1.0))

plt.xlabel('Fraction of edges made undirected')
plt.ylabel('NRMSE')

savefig('undir-performance.svg')
savefig('undir-performance.pdf')
plt.show()

# %%
from notebook import plot_making_edges_undirected
plot_making_edges_undirected()


# %% [markdown]
"""
# Appendix
"""

# %%
%%capture cap

def print_table(df, xheader='Reservoir size N'):
    if ('esn_nrmse', 'mean') not in df.columns:
        print(tabulate(df, headers=[xheader, 'NRMSE'],
                    showindex=False, tablefmt='github'))
        print(f"\nNRMSE mean: {df['esn_nrmse'].mean():.6f}")
        return

    print(tabulate(df, headers=[xheader, 'NRMSE mean', 'NRMSE std'],
                   showindex=False, tablefmt='github'))
    print(f"\nNRMSE pooled mean: {df[('esn_nrmse', 'mean')].mean():.6f}, std: {np.sqrt(np.mean(df[('esn_nrmse', 'std')]**2)):.6f}")

print("# Experiment standard deviations\n")

# Figure 2
df = load_experiment('experiments/lattice_nrmse_is0.1.pkl')
esn = load_experiment('experiments/paper_esn_general_performance.pkl')
esn = esn[esn['hidden_nodes'] <= df['hidden_nodes'].max()]

sq = df[df['w_res_type'] == 'tetragonal']
hex = df[df['w_res_type'] == 'hexagonal']
tri = df[df['w_res_type'] == 'triangular']

agg = { 'esn_nrmse': ['mean', 'std'] }
std_sq = sq.groupby(['hidden_nodes']).agg(agg).reset_index()
std_hex = hex.groupby(['hidden_nodes']).agg(agg).reset_index()
std_tri = tri.groupby(['hidden_nodes']).agg(agg).reset_index()
std_esn = esn.groupby(['hidden_nodes']).agg(agg).reset_index()


print("## Figure 2")
print("\n### Square\n")
print_table(std_sq)
print("\n### Hexagonal\n")
print_table(std_hex)
print("\n### Triangular\n")
print_table(std_tri)
print("\n### ESN\n")
print_table(std_esn)

# Figure 3a
df = load_experiment('experiments/lattice_dir_frac.pkl')

sq = df[df['w_res_type'] == 'tetragonal']
hex = df[df['w_res_type'] == 'hexagonal']
tri = df[df['w_res_type'] == 'triangular']

agg = { 'esn_nrmse': ['mean', 'std'] }
std_sq = sq.groupby(['dir_frac']).agg(agg).reset_index()
std_hex = hex.groupby(['dir_frac']).agg(agg).reset_index()
std_tri = tri.groupby(['dir_frac']).agg(agg).reset_index()

print("\n## Figure 3a")
print("\n### Square\n")
print_table(std_sq, xheader='Fraction of directed edges')
print("\n### Hexagonal\n")
print_table(std_hex, xheader='Fraction of directed edges')
print("\n### Triangular\n")
print_table(std_tri, xheader='Fraction of directed edges')

# Figure 3b
df = load_experiment('experiments/lattice_dir_full.pkl')
esn = load_experiment('experiments/paper_esn_general_performance.pkl')

sq = df[df['w_res_type'] == 'tetragonal']
hex = df[df['w_res_type'] == 'hexagonal']
tri = df[df['w_res_type'] == 'triangular']

agg = { 'esn_nrmse': ['mean', 'std'] }
std_sq = sq.groupby(['hidden_nodes']).agg(agg).reset_index()
std_hex = hex.groupby(['hidden_nodes']).agg(agg).reset_index()
std_tri = tri.groupby(['hidden_nodes']).agg(agg).reset_index()
std_esn = esn.groupby(['hidden_nodes']).agg(agg).reset_index()

print("\n## Figure 3b")
print("\n### Square\n")
print_table(std_sq)
print("\n### Hexagonal\n")
print_table(std_hex)
print("\n### Triangular\n")
print_table(std_tri)
print("\n### ESN\n")
print_table(std_esn)

# Figure 4
sq_local = load_experiment('experiments/lattice_dir_full.pkl')
esn_local = load_experiment('experiments/paper_esn_general_performance.pkl')
sq_global = load_experiment('experiments/paper_directed_lattice_performance.pkl')

sq_local = sq_local[sq_local['w_res_type'] == 'tetragonal']

agg = { 'esn_nrmse': ['mean', 'std'] }
std_sq_local = sq_local.groupby(['hidden_nodes']).agg(agg).reset_index()
std_esn_local = esn_local.groupby(['hidden_nodes']).agg(agg).reset_index()
std_sq_global = sq_global.groupby(['hidden_nodes']).agg(agg).reset_index()

print("\n## Figure 4")
print("\n### Square (local)\n")
print_table(std_sq_local)
print("\n### Square (global)\n")
print_table(std_sq_global)
print("\n### ESN\n")
print_table(std_esn_local)

# Figure 5
esn_shrink_nrmses = pickle.load(open('experiments/paper_esn_removed_nodes_nrmses.pkl', 'rb'))
lattice_shrink_nrmses = pickle.load(open('experiments/deg_lattice_nrmses.pkl', 'rb'))
x = list(range(len(esn_shrink_nrmses), 0, -1))

esn_shrink = pd.DataFrame({'x': x, 'esn_nrmse': esn_shrink_nrmses})
lattice_shrink = pd.DataFrame({'x': x, 'esn_nrmse': lattice_shrink_nrmses})

esn_general = load_experiment('experiments/paper_esn_general_short.pkl')

agg = { 'esn_nrmse': ['mean', 'std'] }
std_esn = esn_general.groupby(['hidden_nodes']).agg(agg).reset_index()
std_esn = std_esn[::-1]

print("\n## Figure 5")

print("\n### Square\n")
print_table(lattice_shrink)

print("\n### ESN (shrinking)\n")
print_table(esn_shrink)

print("\n### Random ESN\n")
print_table(std_esn)

# Figure 7
nrmses = pickle.load(open('experiments/grow_mid_nrmses.pkl', 'rb'))
x = list(range(74, 74+len(nrmses)))
df = pd.DataFrame({'x': x, 'esn_nrmse': nrmses})

print("\n## Figure 7\n")
print_table(df)

# Figure 9
nrmses = pickle.load(open('experiments/changed_edges_nrmses.pkl', 'rb'))
x = [n/264 for n in range(1, 265)]
df = pd.DataFrame({'x': x, 'esn_nrmse': nrmses})

print("\n## Figure 9\n")
print_table(df, xheader='Fraction of directed edges')

with open('../paper/appendix.md', 'w') as file:
    file.write(cap.stdout)
