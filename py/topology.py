from collections import OrderedDict

import dataset as ds
from ESN import Distribution, ESN
from metric import esn_nrmse, evaluate_esn, kernel_quality, memory_capacity
from gridsearch import experiment
from matrix import euclidean


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dataset', action='store_true')
    parser.add_argument('--load_dataset', action='store_true')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.load_dataset:
        dataset = ds.load_dataset()
    else:
        u_train, y_train = ds.NARMA(sample_len = 2000)
        u_test, y_test = ds.NARMA(sample_len = 3000)
        ds.dataset = [u_train, y_train, u_test, y_test]
        if args.cache_dataset:
            ds.cache_dataset(dataset)

    hidden_nodes = 49
    params = {
        'hidden_nodes': hidden_nodes,
        'input_scaling': 1.0,
        'spectral_radius': 0.9,
        'w_res_type': 'waxman',
        'readout': 'rr',
    }

    print('Statistics')

    esn = ESN(**params)
    nrmse = evaluate_esn(ds.dataset, esn)
    print('  NRMSE:\t\t', nrmse)

    inputs = ds.dataset[0]
    ks = esn.hidden_nodes
    kq = kernel_quality(inputs, esn, ks=ks)
    print()
    print('  k_len:\t\t', len(inputs) // ks)
    print('  ks:\t\t\t', ks)
    print('  Kernel quality:\t', kq)
    print('  Reservoir size:\t', esn.hidden_nodes)

    mc = memory_capacity(esn)
    print()
    print('  Memory capacity:\t', mc)


if __name__ == '__main__':
    main()
