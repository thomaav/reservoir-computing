import tqdm
import copy
import pickle
import numpy as np
from collections import OrderedDict

from metric import evaluate_esn
from matrix import find_tetragonal_frontier, find_ways_to_add_node


def remove_nodes_incrementally(dataset, esn, removed_nodes_file):
    try:
        removed_nodes = pickle.load(open(removed_nodes_file, 'rb'))
    except FileNotFoundError:
        removed_nodes = []

    for node in removed_nodes:
        esn.remove_hidden_node(node)

    while esn.hidden_nodes > 1:
        default_nrmse = evaluate_esn(dataset, esn)
        nrmse_diffs = []

        for i in tqdm.tqdm(range(esn.hidden_nodes)):
            _esn = copy.deepcopy(esn)
            _esn.remove_hidden_node(i)
            nrmse = evaluate_esn(dataset, _esn)
            nrmse_diffs.append(nrmse - default_nrmse)

        best_node = np.argmin(nrmse_diffs)
        esn.remove_hidden_node(best_node)
        removed_nodes.append(best_node)

        print()
        best_nrmse = default_nrmse + nrmse_diffs[best_node]
        print(f'it: nrmse-{best_nrmse}')

        pickle.dump(removed_nodes, open(removed_nodes_file, 'wb'))


def evaluate_incremental_node_removal(dataset, esn_file, removed_nodes_file, return_esns=False):
    esn = pickle.load(open(esn_file, 'rb'))
    removed_nodes = pickle.load(open(removed_nodes_file, 'rb'))

    nrmses = []
    esns = []

    for node in tqdm.tqdm(removed_nodes):
        esn.remove_hidden_node(node)
        nrmses.append(evaluate_esn(dataset, esn))
        esns.append(copy.deepcopy(esn))

    if return_esns:
        return nrmses, esns
    else:
        return nrmses


def make_undirected_incrementally(dataset, esn, changed_edges_file):
    try:
        changed_edges = pickle.load(open(changed_edges_file, 'rb'))
    except FileNotFoundError:
        changed_edges = OrderedDict()

    original_edges = [edge for edge in esn.G.edges]

    for edge in changed_edges:
        esn.make_edge_undirected(edge)

    while len(changed_edges) < len(original_edges)*2:
        default_nrmse = evaluate_esn(dataset, esn)
        nrmse_diffs = []

        for i, edge in enumerate(tqdm.tqdm(original_edges)):
            if edge in changed_edges:
                nrmse_diffs.append(np.inf)
                continue

            _esn = copy.deepcopy(esn)
            _esn.make_edge_undirected(edge)
            nrmse = evaluate_esn(dataset, _esn)
            nrmse_diffs.append(nrmse - default_nrmse)

        best_edge = np.argmin(nrmse_diffs)
        edge_to_change = list(original_edges)[best_edge]
        esn.make_edge_undirected(edge_to_change)

        changed_edges[edge_to_change] = None
        changed_edges[(edge_to_change[1], edge_to_change[0])] = None

        print()
        best_nrmse = default_nrmse + nrmse_diffs[best_edge]
        print(f'it: removed-{best_edge}, nrmse-{best_nrmse}, total-{len(changed_edges)//2}')

        pickle.dump(changed_edges, open(changed_edges_file, 'wb'))


def evaluate_incremental_undirection(dataset, esn, changed_edges_file, esns=False):
    changed_edges = pickle.load(open(changed_edges_file, 'rb'))
    changed_edges = [e for i, e in enumerate(changed_edges) if i%2 == 0]

    nrmses = []
    esns = []

    for edge in tqdm.tqdm(changed_edges):
        esn.make_edge_undirected(edge)
        nrmses.append(evaluate_esn(dataset, esn))
        esns.append(copy.deepcopy(esn))

    if esns:
        return nrmses, esns
    else:
        return nrmses


def find_best_node_to_remove(dataset, esn):
    best_nrmse = np.inf
    best_node = None

    for i in tqdm.tqdm(range(esn.hidden_nodes)):
        _esn = copy.deepcopy(esn)
        _esn.remove_hidden_node(i)
        nrmse = evaluate_esn(dataset, _esn)

        if nrmse < best_nrmse:
            best_nrmse = nrmse
            best_node = i

    return i


def find_best_node_to_add(dataset, esn):
    frontier = find_tetragonal_frontier(esn.G)
    possible_edges = {}

    for node in frontier:
        possible_edges[node] = find_ways_to_add_node(esn.G, node)

    best_nrmse = np.inf
    best_node_and_edges = None

    for node in tqdm.tqdm(possible_edges.keys()):
        for edges in possible_edges[node]:
            _esn = copy.deepcopy(esn)
            _esn.add_hidden_node(node, edges)
            nrmse = evaluate_esn(dataset, _esn)

            if nrmse < best_nrmse:
                best_nrmse = nrmse
                best_node_and_edges = (node, edges)

    return best_node_and_edges
