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


def evaluate_esn(dataset, esn):
    u_train, y_train, u_test, y_test = dataset
    esn(u_train, y_train)

    y_predicted = esn(u_test)
    _nmse = nmse(y_predicted, y_test[200:])
    _nrmse = nrmse(y_predicted, y_test[200:])

    return _nrmse


def evaluate_esn_input_sparsity(dataset, hidden_nodes, w_in_sparsity):
    esn = ESN(hidden_nodes=hidden_nodes, w_in_sparsity=w_in_sparsity)
    return evaluate_esn(dataset, esn)


def evaluate_esn_output_sparsity(dataset, hidden_nodes, w_out_sparsity):
    esn = ESN(hidden_nodes=hidden_nodes, w_out_sparsity=w_out_sparsity)
    return evaluate_esn(dataset, esn)


def evaluate_prediction(y_predicted, y):
    plt.plot(y, 'black', linestyle='dashed')
    plt.plot(y_predicted, 'green')
    plt.show()
