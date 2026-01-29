import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#1
data_nat=pd.read_csv("NatLocMovement.csv", index_col=0)
data_pop=pd.read_csv("PopulationLoc.csv", index_col=0)

data_nat.apply(
    lambda col : col.fillna(col.mean()) if col.dtype!='object' else col
)

data_pop.apply(
    lambda col : col.fillna(col.mean()) if col.dtype!='object' else col
)

labels= list(data_nat.columns.values[1:])

merged= data_nat.merge(
    data_pop,
    right_index=True,
    left_index=True
)[['City', 'CountyCode', 'Population'] + labels]

merged["InfantMR"]= merged["DeceasedUnder1Year"]/merged["LiveBirths"] *100
merged[["City","InfantMR"]].to_csv("Req1.csv")

#2
for col in labels:
    merged[col]= merged[col]/(merged['Population']/1000)

merged=merged.set_index('City')

rates= merged.groupby('CountyCode')[labels].idxmax()
rates.to_csv("Req2.csv")

#3
raw_data= pd.read_csv("Dataset_83.csv", index_col=0)

data= raw_data.select_dtypes(include= [np.number])

labels= list(data.columns.values)

x = StandardScaler().fit_transform(data)

cov= np.cov(x, rowvar=False)

cov_df= pd.DataFrame(cov, index=labels, columns=labels)
cov_df.to_csv("Req3.csv")

#4
rows= list(data.index.values)

pca =PCA()
C = pca.fit_transform(x)
pd.DataFrame(
     np.round(C, 2),
     index= rows,
     columns=['C'+str(i+1) for i in range(C.shape[1])]
).to_csv('Req4.csv')

#5
alpha = pca.explained_variance_

plt.figure(figsize=(8,8))
plt.title('Variance explained by the PCs')
Xindex = ['C'+ str(k+1) for k in range(len(alpha))]
plt.plot(Xindex, alpha, 'bo-')
plt.axhline(1, color='r')
plt.show()

#6
loadings = pca.components_.T * np.sqrt(alpha)

plt.figure(figsize=(8,8))
plt.title("Factor loadings")
T= [t for t in np.arange(0, np.pi*2, 0.01)]
X= [np.cos(t) for t in T]
Y= [np.sin(t) for t in T]
plt.plot(X,Y)
plt.axhline(0, c='g')
plt.axvline(0, c='g')
plt.scatter(loadings[:,0], loadings[:,1])
plt.show()