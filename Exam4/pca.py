import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

raw_macro = pd.read_csv("./dataIN/GlobalIndicatorsPerCapita_2021.csv")
raw_cont = pd.read_csv("./dataIN/CountryContinents.csv")

# variabile numerice
labels = raw_macro.columns.values[2:].tolist()

# merge + agregare
merged = raw_macro.merge(raw_cont, left_index=True, right_index=True)
merged = merged.groupby("CountryId").sum()

# Value Added
va_cols = raw_macro.columns.values[9:].tolist()
merged["Value Added"] = merged[va_cols].sum(axis=1)

# eliminare coloane inutile
merged = merged.drop([
    'GNI', 'ChangesInv', 'Exports', 'Imports', 'FinalConsExp',
    'GrossCF', 'HouseholdConsExp', 'AgrHuntForFish',
    'Construction', 'Manufacturing', 'MiningManUt',
    'TradeT', 'TransportComm', 'Other',
    'CountryID', 'Country_y', 'Continent'
], axis=1)

merged.to_csv("./dataOUT/Cerinta1.csv")

#2
merged_std = raw_macro.merge(raw_cont, left_index=True, right_index=True)
merged_mean = raw_macro.merge(raw_cont, left_index=True, right_index=True)

merged_std = merged_std.groupby("Continent").sum()
merged_mean = merged_mean.groupby("Continent").sum()

merged_std = merged_std.drop(['Country_y', 'CountryId', 'Country_x', 'CountryID'], axis=1)
merged_mean = merged_mean.drop(['Country_y', 'Country_x', 'CountryId'], axis=1)

merged_std[labels] = merged_std[labels].std(axis=0)
merged_mean[labels] = merged_mean[labels].mean(axis=0)

merged_std[labels] = merged_std[labels] / merged_mean[labels]

merged_std.to_csv("./dataOUT/Cerinta2.csv")

#3
raw_pca = raw_macro[labels]

X_std = StandardScaler().fit_transform(raw_pca)

pca = PCA()
PC = pca.fit_transform(X_std)

eigenvalues = pca.explained_variance_

eigen_df = pd.DataFrame(
    eigenvalues,
    index=['PCA'+str(i+1) for i in range(len(eigenvalues))]
)

eigen_df.to_csv("./dataOUT/eigenvalues.csv")

#4
scores = PC / np.sqrt(eigenvalues)

scores_df = pd.DataFrame(
    scores,
    index=raw_pca.index,
    columns=raw_pca.columns
)

scores_df.to_csv("./dataOUT/scoruri.csv")

plt.figure(figsize=(8,8))
plt.scatter(scores_df.iloc[:,0], scores_df.iloc[:,1],
            marker="*", color="purple")

for i, label in enumerate(scores_df.index):
    plt.text(scores_df.iloc[i,0], scores_df.iloc[i,1], label, fontsize=8)

plt.axhline(0, linestyle="--", color="gray")
plt.axvline(0, linestyle="--", color="gray")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("PCA Score Plot")
plt.grid()
plt.show()

#5
raw_fact = pd.read_csv("./dataIN/g20.csv", index_col=0)

communalities = np.cumsum(raw_fact * raw_fact, axis=1)

most_important = communalities.sum().idxmax()

print("Cel mai important factor este:", most_important)
