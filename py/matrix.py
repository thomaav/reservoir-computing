import networkx as nx
import numpy as np
from itertools import combinations
from math import sqrt, exp


def euclidean(x, y):
    """
    The euclidean distance metric that is used within NetworkX.
    """
    return sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))


def waxman(n, alpha, beta, connectivity='default', z_frac=1.0, scale=1.0,
           directed=False, sign_frac=0.0, dist_function=euclidean):
    """
    B. M. Waxman, Routing of multipoint connections.

    Adapted from the NetworkX implementation.
    """
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))

    domain = (0, 0, 0, 1, 1, 1)
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

    return G


def tetragonal(dim, periodic=False):
    # (TODO): rectangular
    G = nx.grid_graph(dim, periodic=periodic)

    pos = dict(zip(G, G))
    nx.set_node_attributes(G, pos, 'pos')

    return G


def rectangular(m, n, rect_ratio=1.0, periodic=False):
    G = tetragonal([m, n], periodic=periodic)

    for n in G:
        pos = G.nodes[n]['pos']
        G.nodes[n]['pos'] = (pos[0], pos[1]*rect_ratio)

    return G


def hexagonal(m, n, periodic=False):
    G = nx.hexagonal_lattice_graph(m, n, periodic=periodic)
    return G


def triangular(m, n, periodic=False):
    G = nx.triangular_lattice_graph(m, n, periodic=periodic)
    return G


if __name__ == '__main__':
    G = triangular(5, 5)

    import matplotlib.pyplot as plt
    nx.draw(G, pos=nx.spring_layout(G), with_labels=True)
    plt.show()

