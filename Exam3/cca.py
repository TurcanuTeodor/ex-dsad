import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA

raw_industries= pd.read_csv("./dataIN/Industrie.csv", index_col=0)
raw_pop= pd.read_csv("./dataIN/PopulatieLocalitati.csv", index_col=0)

labels= list(raw_industries.columns.values[1:])
print(labels)

merged= raw_industries.merge(
    raw_pop,
    right_index=True,
    left_index=True
)[['Judet','Populatie']+labels]

print(merged)

#1
ex1= merged.copy()
ex1[labels]= ex1[labels].div(ex1['Populatie'], axis=0)

ex1= ex1.reset_index()
ex1.to_csv('./dataOUT/Req1.csv', index=False)

#2
county=(
    merged
    .groupby('Judet')[labels]
    .sum()
)

dominant_activity= county.idxmax(axis=1)
dominant_value= county.max(axis=1)

ex2= pd.DataFrame({
    'Judet':county.index,
    'Activitate_dominant':dominant_activity,
    'Turnover':dominant_value
})
ex2.to_csv('./dataOUT/ex2.csv', index=False)

#3
raw_meat= pd.read_csv("./dataIN/DataSet_34.csv", index_col=0)

X= raw_meat.iloc[:, 0:4]
Y= raw_meat.iloc[:, 4:8]

X_std= StandardScaler().fit_transform(X)
Y_std= StandardScaler().fit_transform(Y)
#print(X_std)

Xstd= pd.DataFrame(
    X_std,
    index=raw_meat.index,
    columns=X.columns
)
print(Xstd)

Ystd= pd.DataFrame(
    Y_std,
    index=raw_meat.index,
    columns=Y.columns
)

#4
cca= CCA(n_components=2)
cca.fit(Xstd, Ystd)

Z, U= cca.transform(Xstd, Ystd) #derived canonical variables

Z_df= pd.DataFrame(Z, index= Xstd.index, columns=['z1','z2']) # lin combo of X's vars
print(Z_df)
U_df= pd.DataFrame(U, index=Ystd.index, columns=['u1','u2']) # lin combo of Y's vars
print(U_df)

#5 factor/canonical loadings= imporanta vars initiale in formarea componentelor canonice
Rxz= pd.DataFrame(
    cca.x_loadings_,
    index=X.columns,
    columns=['z1','z2']
) #how much does the vars from X contribute to the canonical scores from Z

Ryu= pd.DataFrame(
    cca.y_loadings_,
    index=Y.columns,
    columns=['u1','u2']
) #how much does the vars from Y contribute to the canonical scores from U

Rxz.to_csv("./dataOUT/Rxz.csv")
Ryu.to_csv("./dataOUT/Ryu.csv")

#6
plt.figure(figsize=(7,7))

plt.scatter(Z_df['z1'], Z_df['z2'], label='Z scores', color='purple')
plt.scatter(U_df['u1'], U_df['u2'], label='U scores', color='plum')

plt.axhline(0, color='grey', linestyle='--')
plt.axvline(0, color='grey', linestyle='--')

plt.xlabel('Component1')
plt.ylabel('Component2')
plt.title('CCA biplot')
plt.legend()
plt.grid()
plt.show()