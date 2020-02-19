from ESN import Distribution
from dataset import NARMA, cache_dataset, load_dataset
from metric import eval_esn_with_params, kernel_quality, memory_capacity


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
        dataset = load_dataset()
    else:
        u_train, y_train = NARMA(sample_len = 2000)
        u_test, y_test = NARMA(sample_len = 3000)
        dataset = [u_train, y_train, u_test, y_test]
        if args.cache_dataset:
            cache_dataset(dataset)

    hidden_nodes = 50
    params = {
        'hidden_nodes': hidden_nodes,
        'input_scaling': 0.9,
        'w_res_type': 'waxman'
    }

    print('Statistics')

    nrmse, esn = eval_esn_with_params(dataset, params=params)
    print('  NRMSE:\t\t', nrmse)

    inputs = dataset[0]
    ks = hidden_nodes
    kq = kernel_quality(inputs, esn, ks=ks)
    print()
    print('  k_len:\t\t', len(inputs) // ks)
    print('  ks:\t\t\t', ks)
    print('  Kernel quality:\t', kq)
    print('  Reservoir size:\t', hidden_nodes)

    mc = memory_capacity(esn)
    print()
    print('  Memory capacity:\t', mc)


if __name__ == '__main__':
    main()
