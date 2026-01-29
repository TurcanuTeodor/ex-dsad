import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

raw_budget= pd.read_csv("...", index_col=0)
raw_pop= pd.read_csv("...", index_col=0)

revenues= [f"V{i}" for i in range(1,6)] #v1..v5
expenses= [f"C{i}" for i in range(1,9)] #c1..c8

merged = raw_budget.merge(raw_pop, left_index=True, right_index=True)

pop_col= 'Population'
county_col= 'CountyCode'

cols_needed= ['City', pop_col, county_col] + revenues +expenses
merged= merged[cols_needed]
print(merged.head())

#1
ex1= merged.copy()

ex1[revenues]= ex1[revenues].div(ex1[pop_col], axis=0)
ex1[expenses]= ex1[expenses].div(ex1[pop_col], axis=0)

ex1_out= ex1.reset_index()[['index', 'City']+ revenues+expenses]
ex1_out= ex1.rename(columns={'index':'Siruta'})

ex1_out.to_csv("...", index=False)

#2
county_exp=(
    merged
    .groupby(county_col)[expenses]
    .sum()
)

total_exp= county_exp.sum(axis=1)
perc= county_exp.div(total_exp, axis=0) *100

ex2_out= perc.reset_index()
ex2_out.to_csv("...", index=False)

#3
raw_df= pd.read_csv('...', index_col=0)
df_cols= raw_df.columns.values.tolist()

scaler=StandardScaler()
X_std= scaler.fit_transform(raw_df[df_cols])

Xstd_df= pd.DataFrame(
    X_std, 
    index= raw_df.index, 
    columns=df_cols
)

cov_matrix= Xstd_df.cov()
cov_matrix.to_csv("...")

#4
pca= PCA()
PC= pca.fit_transform(X_std)

alpha= pca.explained_variance_
pve= pca.explained_variance_ratio_

scores= PC/ np.sqrt(alpha)
scores_df= pd.DataFrame(
    scores,
    index= raw_df.index,
    columns=[f"PC{i+1}" for i in range(scores.shape[1])]
)

scores_df.to_csv('...')
