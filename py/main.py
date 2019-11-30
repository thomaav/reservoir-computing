import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from metric import eval_esn_with_params
from dataset import NARMA
from plot import *
from ESN import Distribution


def run_single_esn(dataset):
    params = {
        'hidden_nodes': 200,
    }

    nrmse, esn = eval_esn_with_params(dataset, params=params)
    print('NRMSE:', nrmse)


def main():
    u_train, y_train = NARMA(sample_len = 2000)
    u_test, y_test = NARMA(sample_len = 3000)
    dataset = [u_train, y_train, u_test, y_test]

    import argparse
    parser = argparse.ArgumentParser(description='tms RC')
    parser.add_argument('--input_density', action='store_true')
    parser.add_argument('--output_density', action='store_true')
    parser.add_argument('--output_nodes', action='store_true')
    parser.add_argument('--partial_visibility', action='store_true')
    parser.add_argument('--w_in_distribution', action='store_true')
    parser.add_argument('--w_res_density', action='store_true')
    parser.add_argument('--input_noise', action='store_true')
    parser.add_argument('--input_noise_trained', action='store_true')
    parser.add_argument('--adc_quantization', action='store_true')
    parser.add_argument('--performance', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--single_esn', action='store_true')
    args = parser.parse_args()

    if args.input_density:
        plot_input_density(dataset)
    elif args.output_density:
        plot_output_density(dataset)
    elif args.output_nodes:
        plot_output_nodes(dataset)
    elif args.partial_visibility:
        plot_partial_visibility(dataset)
    elif args.w_in_distribution:
        plot_input_scaling_input_distrib(dataset)
    elif args.w_res_density:
        plot_w_res_density_w_res_distrib(dataset)
    elif args.input_noise:
        plot_input_noise(dataset)
    elif args.input_noise_trained:
        plot_input_noise_trained(dataset)
    elif args.adc_quantization:
        plot_adc_quantization(dataset)
    elif args.performance:
        performance_sweep(dataset)
    elif args.visualize:
        visualize(dataset)
    elif args.single_esn:
        run_single_esn(dataset)


if __name__ == '__main__':
    main()
