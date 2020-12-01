import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.graph import graph_shortest_path
def isomap(data, component=2, neighbor=50):
    data= calculate_dist(data, neighbor)
    graph = graph_shortest_path(data, directed=False)
    graph = -0.5 * (graph ** 2)
    return MDS(graph, component)

def calculate_dist(X, neighbor=6):
    D = np.array([[np.sqrt(sum((p2 - p1)**2)) for p2 in X] for p1 in X])
    neighbors = np.zeros_like(D)
    sort_distances = np.argsort(D, axis=1)[:, 1:neighbor+1]
    for k,i in enumerate(sort_distances):
        neighbors[k,i] = D[k,i]
    #print(neighbors)
    return neighbors

def centering(matrix):
    N = matrix.shape[0]
    mean_rows = np.sum(matrix, axis=0) / N
    mean_cols = (np.sum(matrix, axis=1)/N)[:, np.newaxis]
    mean_all = meanrows.sum() / N
    matrix -= mean_rows
    matrix -= mean_cols
    matrix+= mean_all
    return matrix

def MDS(data, component=2):
    centering(data)
    v, d = np.linalg.eig(data)
    U = [(np.abs(v[i]), d[:, i]) for i in range(len(v))]
    U.sort(key=lambda x: x[0], reverse=True)
    U = np.array(U)
    matrix_w = np.hstack([U[i, 1].reshape(data.shape[1], 1) for i in range(component)])
    return matrix_w

row=101
col=16
X = np.zeros([row, col])
T = np.zeros(row)

f = open("zoo.data","r")

for index, line in enumerate(f):
    dims = line.split(",")
    X[index] = dims[1:17]
    T[index] = dims[17]

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)

result=isomap(X)
plt.figure()
plt.scatter(result[:,0], result[:,1], c = T)

plt.show()