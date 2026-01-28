import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#std
X_std= StandardScaler().fit_transform(X)

#apply
pca= PCA()
C= pca.fit_transform(X_std) #scores

#Eigenvalues & explained variance
alpha= pca.explained_variance_  #eigenvalues
pve= pca.explained_variance_ratio_  #% variance explained

#Loadings (correlation X – PC)
loadings= pca.components_.T * np.sqrt(alpha)

#Communalities
communalities =np.cumsum(loadings**2, axis=1)

#scores
scores_df= pd.DataFrame(
    C,
    index= df.index,
    columns=[f"PC{i+1}" for i in range(C.shape[1])]
)
scores_df.to_csv("scores.csv")

#scree plt
plt.bar(range(1,len(pve)+1), pve)
plt.xlabel("PC")
plt.ylabel("Variance Expl")
plt.title("Scree plot")
plt.show()