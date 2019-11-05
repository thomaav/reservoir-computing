import matplotlib.pyplot as plt
import networkx as nx

from ESN import *


if __name__ == '__main__':
    # Fetch an adjacency matrix.
    esn = ESN(hidden_nodes=20, w_res_density=0.1)
    A = esn.w_res.numpy()

    # Create a network graph from the ESN network.
    G = nx.from_numpy_matrix(A)
    nx.draw(G, pos=nx.spring_layout(G), with_labels=True)
    plt.show()
