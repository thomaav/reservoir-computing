import networkx as nx
import numpy as np
import torch
from itertools import combinations, product
from math import sqrt, exp
from collections import OrderedDict


def euclidean(x, y):
    """
    The euclidean distance metric that is used within NetworkX.
    """
    return sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))


def inv(x, y): return 1/euclidean(x, y)
def inv_squared(x, y): return 1/euclidean(x, y)**2
def inv_cubed(x, y): return 1/euclidean(x, y)**3


def waxman(n, alpha, beta, connectivity='default', z_frac=1.0, scale=1.0,
           directed=False, sign_frac=0.0, dist_function=euclidean, l=0,
           dim_size=1):
    """
    B. M. Waxman, Routing of multipoint connections.

    Adapted from the NetworkX implementation.
    """
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))

    domain = (0, 0, 0, dim_size, dim_size, dim_size)
    xmin, ymin, zmin, xmax, ymax, zmax = domain

    uniform = np.random.uniform
    def z(): return uniform(zmin, zmax) if np.random.random() < z_frac else 0.0
    pos = {v: (uniform(xmin, xmax), uniform(ymin, ymax), z()) for v in G}
    nx.set_node_attributes(G, pos, 'pos')

    # L is the maximum distance between any pair of nodes, but can be extended
    # to always be set to a specific value.
    L = max(euclidean(x, y) for x, y in combinations(pos.values(), 2))
    def dist(u, v): return euclidean(pos[u], pos[v])

    def should_join(pair):
        return np.random.random() < beta*exp(-dist(*pair) / (alpha*L))

    if connectivity == 'default':
        G.add_edges_from(filter(should_join, combinations(G, 2)))
    elif connectivity == 'global':
        for pair in combinations(G, 2):
            u, v = pair[0], pair[1]
            weight_sign = -1 if np.random.random() < sign_frac else 1
            if directed:
                u, v = (u, v) if np.random.random() < .5 else (v, u)
                G.add_edge(u, v, weight=weight_sign*dist_function(pos[u], pos[v])*scale)
            else:
                G.add_edge(u, v, weight=weight_sign*dist_function(pos[u], pos[v])*scale)
                G.add_edge(v, u, weight=weight_sign*dist_function(pos[v], pos[u])*scale)
    elif connectivity == 'rgg_example':
        for pair in combinations(G, 2):
            u, v = pair[0], pair[1]
            if dist_function(pos[u], pos[v]) < 0.10:
                G.add_edge(u, v, weight=1)
                G.add_edge(v, u, weight=1)

    return G


def add_edge(G, u, v, dist_function=euclidean):
    if u != v:
        G.add_edge(u, v, weight=dist_function(G.nodes[u]['pos'], G.nodes[v]['pos']))


def tetragonal(dim, periodic=False, dist_function=None):
    # (TODO): rectangular
    G = nx.grid_graph(dim, periodic=periodic)

    pos = dict(zip(G, G))
    nx.set_node_attributes(G, pos, 'pos')

    return G


def rectangular(m, n, rect_ratio=1.0, periodic=False, dist_function=None):
    G = tetragonal([m, n], periodic=periodic)

    for n in G:
        pos = G.nodes[n]['pos']
        G.nodes[n]['pos'] = (pos[0], pos[1]*rect_ratio)

    for u, v, d in G.edges(data=True):
        d['weight'] = 1/euclidean(G.nodes[u]['pos'], G.nodes[v]['pos'])

    return G


def hexagonal(m, n, periodic=False, dist_function=None):
    G = nx.hexagonal_lattice_graph(m, n, periodic=periodic)
    return G


def triangular(m, n, periodic=False, dist_function=None):
    G = nx.triangular_lattice_graph(m, n, periodic=periodic)
    return G


def grow_neighborhoods(G, dist_function=euclidean, l=1):
    def get_neighbors(A, n): return A[n].nonzero()[0]

    # I have no idea what is happening with the np matrix here, as there seems
    # to be some sort of infinite recurrence of the first element -- but this
    # does not happen with torch? What in the world is going on…
    A = nx.to_numpy_matrix(G)
    A = torch.FloatTensor(A).data.numpy()

    # Required to get the correct u,v for add_edge.
    nodes = {i: n for i, n in enumerate(G.nodes())}

    for n in range(len(A)):
        cur_neigh = get_neighbors(A, n)

        for i in range(l):
            next_neigh = np.concatenate([get_neighbors(A, neigh) for neigh in cur_neigh])
            next_neigh = np.setdiff1d(next_neigh, cur_neigh)
            cur_neigh = np.concatenate([cur_neigh, next_neigh])

        # We are not modifying A by doing this, so we can add the edges directly
        # without worrying about growing the neighborhood indefinitely.
        new_neigh = cur_neigh
        for neigh in new_neigh:
            u = nodes[n]
            v = nodes[neigh]
            add_edge(G, u, v, dist_function)


def make_weights_negative(G, sign_frac):
    for u, v, d in G.edges(data=True):
        sign = -1 if np.random.random() < sign_frac else 1
        d['weight'] = d['weight']*sign if 'weight' in d else sign


def make_graph_directed(G, dir_frac):
    bidir_edges = G.edges()
    dir_G =  G.to_directed()

    for u,v in bidir_edges:
        if np.random.random() < dir_frac:
            del_u, del_v = (u,v) if np.random.random() < 0.5 else (v,u)
            dir_G.remove_edge(del_u, del_v)

    return dir_G


def find_tetragonal_frontier(G):
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    existing_nodes = set([n[1]['pos'] for n in G.nodes(data=True)])
    frontier = set()

    for node in existing_nodes:
        for dir in dirs:
            pos = (node[0]+dir[0], node[1]+dir[1])
            if pos in existing_nodes:
                continue
            frontier.add(pos)

    return frontier


def find_ways_to_add_node(G, node):
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    existing_nodes = set([n[1]['pos'] for n in G.nodes(data=True)])
    neighbors = OrderedDict()

    for dir in dirs:
        pos = (node[0]+dir[0], node[1]+dir[1])
        if pos in existing_nodes:
            neighbors[pos] = None

    to_edges = [(node, neigh) for neigh in neighbors]
    from_edges = [(neigh, node) for neigh in neighbors]
    dir = [to_edges, from_edges]

    # ...
    ways_to_add = []
    edge_combinations = ["".join(seq) for seq in product("01", repeat=len(neighbors))]
    for choice in edge_combinations:
        edges_to_add = [dir[int(c)][i] for i, c in enumerate(choice)]
        ways_to_add.append(edges_to_add)

    return ways_to_add


if __name__ == '__main__':
    G = triangular(5, 5)

    import matplotlib.pyplot as plt
    nx.draw(G, pos=nx.spring_layout(G), with_labels=True)
    plt.show()

