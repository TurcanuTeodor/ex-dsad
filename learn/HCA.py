import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


X = df[X_cols]

#std
X_std = StandardScaler().fit_transform(X)

#linkage ward
Z = linkage(X_std, method="ward")  # matricea ierarhiei (linkage matrix)
Z_df = pd.DataFrame(Z, columns=["c1", "c2", "dist", "new_cluster_size"])
Z_df.to_csv("./dataOUT/hca_linkage_matrix.csv", index=False)

#dendograma
plt.figure(figsize=(10,6))
dendrogram(Z, labels=df.index.astype(str).values)
plt.title("Dendrogram (Ward)")
plt.xlabel("Observations")
plt.ylabel("Distance")
plt.grid()
plt.show()

#partition
k = 3
clusters = fcluster(Z, t=k, criterion="maxclust")

partition_df = pd.DataFrame({"Cluster": clusters}, index=df.index)
partition_df.to_csv("./dataOUT/hca_partition.csv")

# mean clusters
df_with_cluster = df.copy()
df_with_cluster["Cluster"] = clusters

cluster_means = df_with_cluster.groupby("Cluster")[X_cols].mean()
cluster_means.to_csv("./dataOUT/hca_cluster_means.csv")

print(cluster_means)
