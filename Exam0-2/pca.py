import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#1------------------------------------------------------
#read each df
data_nat=pd.read_csv("NatLocMovement.csv", index_col=0) #your 1st col name => becomes an index
data_pop=pd.read_csv("PopulationLoc.csv", index_col=0)

#clean df
data_nat.apply(
    lambda col : col.fillna(col.mean()) if col.dtype!='object' else col
)

data_pop.apply(
    lambda col : col.fillna(col.mean()) if col.dtype!='object' else col
)

#take labels of 1 dataset for merging into the other
labels= list(data_nat.columns.values[1:]) # the City col is missed to be later added at the beginning in merge

#merge csvs using the safest approach
#right/left_index = True means match on the same index -> which would be the 1st col that became an index in the start
merged= data_nat.merge(
    data_pop,
    right_index=True,
    left_index=True
)[['City', 'CountyCode', 'Population'] + labels]

#create + calc new col
merged["InfantMR"]= merged["DeceasedUnder1Year"]/merged["LiveBirths"] *100
merged[["City","InfantMR"]].to_csv("Req1.csv") # catch that you need to pass a df [[]] to csv and because SIRUTA is now an index -> will be included by default bc index=True

#2----------------------------------------------------
#for every column get the rate (divide by pop)
for col in labels:
    merged[col]= merged[col]/(merged['Population']/1000)

#set city as index
merged=merged.set_index('City')

#city w max per county
rates= merged.groupby('CountyCode')[labels].idxmax()
rates.to_csv("Req2.csv")

#3-------------------------------------------------------
raw_data= pd.read_csv("Dataset_83.csv", index_col=0)

#only num
data= raw_data.select_dtypes(include= [np.number])

#store labels for labeling the cov matrix
labels= list(data.columns.values)

#standardize = put all the variables on the same scale
#also called a z-score standardization
#! need to std column by column not the whole table
x = StandardScaler().fit_transform(data)

#compute cov matrix of std variables/col
#diagonal is variance(i), because cov(i,i) = var(i)
cov= np.cov(x, rowvar=False) #rowvar=False => col are the variables

#put cov matrix into a df
cov_df= pd.DataFrame(cov, index=labels, columns=labels) #index is also the column because is a variable-by-variable table
cov_df.to_csv("Req3.csv")

#4----------------------------------------------------------------
rows= list(data.index.values)

#create a pca obj
pca =PCA() #object that knows how to find the principal directions of variation
C = pca.fit_transform(x)  #fit => computes cov, eigenvectors(directions of max var) and eigenvalues(how much var each direction explains)
                         #transform => projects each obs onto those directions => produces new var = PC
pd.DataFrame(
     np.round(C, 2), # round to 2 decimal
     index= rows, #label each row w the original id
     columns=['C'+str(i+1) for i in range(C.shape[1])]#these are the principal components C1=PC1 ...
).to_csv('Req4.csv')

#5----------------------------------------------------------------
#find explain variance
alpha = pca.explained_variance_ #array of eigenvalues where 1 eigenvalue per PC

plt.figure(figsize=(8,8))
plt.title('Variance explained by the PCs')
Xindex = ['C'+ str(k+1) for k in range(len(alpha))] #creates C1, C2 for PCs
plt.plot(Xindex, alpha, 'bo-') # draws scree plot
                                     # x axis= PC
                                     # y axis= variance expl/eigenvalues
                                     #bo- means blue points connected by lines
plt.axhline(1, color='r') #draws horizontal reference line at 1( 1 bc data is std, and var =1)
plt.show()

#6-----------------------------------------------------------------
#loadings= eigenvectors * sqr root of eigenvalues
#pca.components_ -> eigenvectors (rows=components, cols=features); .T = transpose(rows=features, cols=components)
#pca.explained_variance_ -> eigenvalues
loadings = pca.components_.T * np.sqrt(alpha)

plt.figure(figsize=(8,8))
plt.title("Factor loadings")
T= [t for t in np.arange(0, np.pi*2, 0.01)] #creates angle values from 0-2pi  used to param a circle, math: (x,y)=(cos t, sin t)
X= [np.cos(t) for t in T]
Y= [np.sin(t) for t in T]
plt.plot(X,Y)
plt.axhline(0, c='g')
plt.axvline(0, c='g')
plt.scatter(loadings[:,0], loadings[:,1]) #":" => all values; 0 =>1st PC, 1=> 2nd PC
plt.show()