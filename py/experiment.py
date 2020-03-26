import tqdm
import copy
import pickle
import numpy as np

from metric import evaluate_esn


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
        print(f'it: removed-{best_node}, nrmse-{nrmse}')

        pickle.dump(removed_nodes, open(removed_nodes_file, 'wb'))


def evaluate_incremental_node_removal(dataset, esn_file, removed_nodes_file, esns=False):
    esn = pickle.load(open(esn_file, 'rb'))
    removed_nodes = pickle.load(open(removed_nodes_file, 'rb'))

    nrmses = []
    esns = []

    for node in tqdm.tqdm(removed_nodes):
        esn.remove_hidden_node(node)
        nrmses.append(evaluate_esn(dataset, esn))
        esns.append(copy.deepcopy(esn))

    if esns:
        return nrmses, esns
    else:
        return nrmses
