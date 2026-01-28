import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler

alcohol = pd.read_csv("./dataIN/alcohol.csv", index_col=0)
countries = pd.read_csv("./dataIN/CoduriTariExtins.csv", index_col=0)

years = ['2000', '2005', '2010', '2015', '2018']

ex1 = alcohol.copy()
ex1['Media'] = ex1[years].mean(axis=1)

result1 = ex1[['Media']]
result1.to_csv("./dataOUT/cerinta1.csv")

print(result1.head())

#2
merged = alcohol.merge(countries, left_index=True, right_index=True)

continent_mean = merged.groupby('Continent')[years].mean()

max_year = continent_mean.idxmax(axis=1)

result2 = pd.DataFrame({
    'Continent_Name': continent_mean.index,
    'Anul': max_year
})

result2.to_csv("./dataOUT/cerinta2.csv", index=False)

print(result2)

#3
X = alcohol[years]

X_std = StandardScaler().fit_transform(X)

Z = linkage(X_std, method='ward')

print(Z)

#4
plt.figure(figsize=(10,6))
dendrogram(Z, labels=alcohol.index.values)
plt.title("Dendrograma – metoda Ward")
plt.xlabel("Țări")
plt.ylabel("Distanță")
plt.grid()
plt.show()

#5
clusters = fcluster(Z, t=3, criterion='maxclust')

partition = pd.DataFrame({
    'Country': alcohol.index,
    'Cluster': clusters
})

partition.to_csv("./dataOUT/pop.csv", index=False)

print(partition.head())
