import matplotlib.pyplot as plt
import numpy as np

from metric import nrmse, nmse, evaluate_prediction
from dataset import NARMA
from ESN import ESN


def main():
    u_train, y_train = NARMA(sample_len = 2000)
    u_test, y_test = NARMA(sample_len = 3000)

    esn = ESN()
    esn(u_train, y_train)

    y_predicted = esn(u_test)
    evaluate_prediction(y_predicted, y_test[200:])
    print('NRMSE:', nrmse(y_predicted, y_test[200:]))
    print('NMSE:', nmse(y_predicted, y_test[200:]))


if __name__ == '__main__':
    main()
