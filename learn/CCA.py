import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA


X = df[X_cols]
Y = df[Y_cols]

#std
X_std = StandardScaler().fit_transform(X)
Y_std = StandardScaler().fit_transform(Y)

Xstd_df = pd.DataFrame(X_std, index=df.index, columns=X_cols)
Ystd_df = pd.DataFrame(Y_std, index=df.index, columns=Y_cols)
Xstd_df.to_csv("./dataOUT/Xstd.csv")
Ystd_df.to_csv("./dataOUT/Ystd.csv")

#apply
cca = CCA(n_components=2)
cca.fit(X_std, Y_std)

Z, U = cca.transform(X_std, Y_std)  # canonical scores

Z_df = pd.DataFrame(Z, index=df.index, columns=["z1","z2"])
U_df = pd.DataFrame(U, index=df.index, columns=["u1","u2"])
Z_df.to_csv("./dataOUT/Xscore.csv")
U_df.to_csv("./dataOUT/Yscore.csv")

#cann correl
canon_corr = []
for k in range(Z.shape[1]):
    corr = np.corrcoef(Z[:, k], U[:, k])[0, 1]
    canon_corr.append(corr)

canon_df = pd.DataFrame(
    {"canonical_correlation": canon_corr},
    index=[f"root{k+1}" for k in range(len(canon_corr))]
)
canon_df.to_csv("./dataOUT/canonical_correlations.csv")

# loadings
Rxz = pd.DataFrame(cca.x_loadings_, index=X_cols, columns=["z1","z2"])
Ryu = pd.DataFrame(cca.y_loadings_, index=Y_cols, columns=["u1","u2"])
Rxz.to_csv("./dataOUT/Rxz.csv")
Ryu.to_csv("./dataOUT/Ryu.csv")

#biplot
plt.figure(figsize=(7,7))
plt.scatter(Z_df["z1"], Z_df["z2"], marker="*", label="Z (X scores)")
plt.scatter(U_df["u1"], U_df["u2"], marker="o", label="U (Y scores)")

for i, label in enumerate(df.index):
    plt.text(Z_df.iloc[i,0], Z_df.iloc[i,1], str(label), fontsize=8)

plt.axhline(0, linestyle="--")
plt.axvline(0, linestyle="--")
plt.xlabel("root 1")
plt.ylabel("root 2")
plt.title("CCA Biplot (Z and U scores)")
plt.grid()
plt.legend()
plt.show()
