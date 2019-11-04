import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from metric import evaluate_esn
from dataset import NARMA
from plot import *


def run_single_esn(dataset):
    esn = ESN(
        hidden_nodes=200,
        input_scaling=2.0,
    )

    print('NRMSE:', evaluate_esn(dataset, esn))


def main():
    u_train, y_train = NARMA(sample_len = 2000)
    u_test, y_test = NARMA(sample_len = 3000)
    dataset = [u_train, y_train, u_test, y_test]

    import argparse
    parser = argparse.ArgumentParser(description='tms RC')
    parser.add_argument('--input_sparsity', action='store_true')
    parser.add_argument('--input_sparsity_scaling', action='store_true')
    parser.add_argument('--output_sparsity', action='store_true')
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--input_noise', action='store_true')
    parser.add_argument('--performance', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--single_esn', action='store_true')
    args = parser.parse_args()

    if args.input_sparsity:
        grid_search_input_sparsity(dataset)
    if args.input_sparsity_scaling:
        grid_search_input_sparsity_input_scaling(dataset)
    elif args.output_sparsity:
        grid_search_output_sparsity(dataset)
    elif args.partial:
        grid_search_partial_visibility(dataset)
    elif args.input_noise:
        input_noise(dataset)
    elif args.performance:
        performance_sweep(dataset)
    elif args.visualize:
        visualize(dataset)
    elif args.single_esn:
        run_single_esn(dataset)


if __name__ == '__main__':
    main()
