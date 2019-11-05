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


def evaluate_esn(dataset, esn, washout=200, plot=False):
    u_train, y_train, u_test, y_test = dataset
    esn(u_train, y_train)

    y_predicted = esn(u_test)
    _nmse = nmse(y_predicted, y_test[washout:])
    _nrmse = nrmse(y_predicted, y_test[washout:])

    if plot:
        target = y_test[washout:]
        predicted = y_predicted

        plt.plot(target, 'black', label='Target output')
        plt.plot(predicted, 'red', label='Predicted output', alpha=0.5)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                   ncol=2, mode="expand", borderaxespad=0., fancybox=False)

        plt.ylabel('Reservoir output')
        plt.xlabel('Time')

        plt.show()

    return _nrmse


def eval_esn_with_params(dataset, params={}):
    esn = ESN(**params)
    return evaluate_esn(dataset, esn), esn


def evaluate_prediction(y_predicted, y):
    plt.plot(y, 'black', linestyle='dashed')
    plt.plot(y_predicted, 'green')
    plt.show()
