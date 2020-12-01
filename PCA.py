import numpy as np
import pandas as pd
from sklearn import decomposition
import matplotlib.pyplot as plt
url="zoo.data"

names=['name', 
'hair',
'feathers',
'eggs',
'milk',
'airborne',
'aquatic',
'predator',
'toothed',
'backbone',
'breathes',
'venomous',
'fins',
'legs',
'tail',
'domestic',
'catsize',
'type']
dataset=pd.read_csv(url,names=names)
y=dataset['name']
S=dataset.drop('name',1)
X=S.drop('type',1)
T=dataset['type']

'''
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.decomposition import PCA

pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print(pca.explained_variance_ratio_)
'''

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)


print(X)
pca = decomposition.PCA(n_components=3)
pca.fit(X)
result = pca.transform(X)

plt.figure()
plt.scatter(result[:,0], result[:,1], c = T)

plt.show()
