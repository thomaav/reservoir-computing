import matplotlib.pyplot as plt
import torch

from ESN import ESN


def nrmse(y_predicted, y):
    var = torch.std(y)**2
    error = (y - y_predicted)**2
    return float(torch.sqrt(torch.mean(error) / var))


def nmse(y_predicted, y):
    var = torch.std(y)**2
    error = (y - y_predicted)**2
    return float(torch.mean(error) / var)


def evaluate_esn(dataset, esn, washout=200):
    u_train, y_train, u_test, y_test = dataset
    esn(u_train, y_train)

    y_predicted = esn(u_test)
    _nmse = nmse(y_predicted, y_test[washout:])
    _nrmse = nrmse(y_predicted, y_test[washout:])

    return _nrmse


def evaluate_esn_input_density(dataset, hidden_nodes, w_in_density):
    esn = ESN(hidden_nodes=hidden_nodes, w_in_density=w_in_density)
    return evaluate_esn(dataset, esn)


def evaluate_esn_output_density(dataset, hidden_nodes, w_out_density):
    esn = ESN(hidden_nodes=hidden_nodes, w_out_density=w_out_density)
    return evaluate_esn(dataset, esn)


def eval_partial_visibility(dataset, w_in_density, w_out_density):
    esn = ESN(hidden_nodes=100, w_in_density=w_in_density, w_out_density=w_out_density)
    return evaluate_esn(dataset, esn)


def evaluate_esn_input_density_scaling(dataset, input_scaling, w_in_density):
    esn = ESN(hidden_nodes=200, input_scaling=input_scaling, w_in_density=w_in_density)
    return evaluate_esn(dataset, esn)


def evaluate_prediction(y_predicted, y):
    plt.plot(y, 'black', linestyle='dashed')
    plt.plot(y_predicted, 'green')
    plt.show()
