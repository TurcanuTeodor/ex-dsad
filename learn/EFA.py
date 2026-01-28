import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis


X = df[X_cols]

#std
X_std = StandardScaler().fit_transform(X)

#apply
m = 2  # nr factori
fa = FactorAnalysis(n_components=m, random_state=0)
F = fa.fit_transform(X_std)  # factor scores (scoruri pe factori)

# loadings
loadings = fa.components_.T

loadings_df = pd.DataFrame(
    loadings,
    index=X_cols,
    columns=[f"F{i+1}" for i in range(m)]
)
loadings_df.to_csv("./dataOUT/efa_loadings.csv")

#communalities
h2 = np.sum(loadings**2, axis=1)
comm_df = pd.DataFrame({"communality_h2": h2}, index=X_cols)
comm_df.to_csv("./dataOUT/efa_communalities.csv")

# factor scores
scores_df = pd.DataFrame(
    F,
    index=df.index,
    columns=[f"F{i+1}" for i in range(m)]
)
scores_df.to_csv("./dataOUT/efa_scores.csv")

#plt
if m >= 2:
    plt.figure(figsize=(7,7))
    plt.scatter(scores_df["F1"], scores_df["F2"], marker="*")
    for i, label in enumerate(scores_df.index):
        plt.text(scores_df.iloc[i,0], scores_df.iloc[i,1], str(label), fontsize=8)
    plt.axhline(0, linestyle="--")
    plt.axvline(0, linestyle="--")
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.title("EFA Score Plot (F1 vs F2)")
    plt.grid()
    plt.show()

#Barplot communalities
plt.figure()
plt.bar(range(len(h2)), h2)
plt.xticks(range(len(h2)), X_cols, rotation=90)
plt.ylabel("Communality (h2)")
plt.title("EFA Communalities")
plt.grid()
plt.tight_layout()
plt.show()
