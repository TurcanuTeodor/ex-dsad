import pandas as pd
import numpy as np
from pandas.core.dtypes.common import is_any_real_numeric_dtype
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

#read
df_nat= pd.read_csv("NatLocMovement.csv", index_col=0)
df_pop= pd.read_csv("PopulationLoc.csv", index_col=0)

#clean
df_nat= df_nat.apply(
    lambda col: col.fillna(col.mean()) if col.dtype!='object' else col
)

df_pop= df_pop.apply(
    lambda col:col.fillna(col.mean()) if col.dtype!='object' else col
)

#take labels for merging
labels= list(df_nat.columns.values[1:])

#merge
merged= df_nat.merge(
    df_pop,
    left_index=True,
    right_index=True
)[['City','CountyCode', 'Population']+labels]

#1 - calc Rate Nat Increase
county=(
    merged
    .groupby('CountyCode', as_index=False)
    .agg({
        'LiveBirths':'sum',
        'Deceased':'sum',
        'Population':'sum'
    })
)

county['Birth_R']= (county['LiveBirths']/county['Population'])*1000
county['Death_R']= (county['Deceased']/county['Population'])*1000
county['Nat_Increase_R']= county['Birth_R']-county['Death_R']

county[['CountyCode', 'Nat_Increase_R']].to_csv('Req1.csv', index=False)

#2 - max rates
city= merged.groupby(['CountyCode','City'], as_index=False).sum()
print(city)

for col in labels:
    city[col]= (city[col]/city['Population'])*1000

result= city.set_index('City').groupby('CountyCode')[labels].idxmax()
result.to_csv('Req2.csv')

#3 - hca
raw_dataset= pd.read_csv("DataSet_34.csv")
data= raw_dataset.select_dtypes(include=[np.number])

x_std= StandardScaler().fit_transform(data) #standardize matrix
pd.DataFrame(x_std, columns=data.columns.values).to_csv("Xstd.csv")

HC= linkage(x_std, method='ward', metric='euclidean') #clustering
print(HC)

#4
#fetch distances
distances= HC[:, 2]

#differences
diff= np.diff(distances)

#max difference position /junction
idx= np.argmax(diff)

#determine threshold
threshold= (distances[idx]+distances[idx+1])/2
print("Junction idx: ", idx)
print("Threshold values: ", threshold)

#5
plt.figure(figsize=(12, 12))
plt.title('Dendogram')
dendrogram(HC, labels=raw_dataset.index.values, leaf_rotation=45)
plt.axhline(threshold, c='r', linestyle='--')
plt.show()