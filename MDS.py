import numpy as np
import pandas as pd
from sklearn import decomposition
import matplotlib.pyplot as plt


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

#print(X)

D = np.zeros([row, row])
for i in range(row):
    for j in range(row):
        value = np.matmul(X[i].T,X[i]) + np.matmul(X[j].T,X[j]) - 2 * np.matmul(X[i].T,X[j])
        D[i][j] = value


one = np.ones([row,1])
S = -1/2 * (D - np.matmul(np.matmul(D, one), one.T)/row - np.matmul(np.matmul(one, one.T), D)/row
    + np.matmul(np.matmul(np.matmul(np.matmul(one, one.T), D), one), one.T)/row**2)


d,v = np.linalg.eig(S)
Diag = np.diag(d)
result = v[:,:2].dot(Diag[:2,:2])


plt.figure()
plt.scatter(result[:,0], result[:,1], c = T)

plt.show()