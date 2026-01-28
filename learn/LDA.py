import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score

X = df[X_cols].values
y = df[y_col].values

#std
X_std = StandardScaler().fit_transform(X)

#apply
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_std, y)  # scores în spațiul discriminant

#scores
scores_df = pd.DataFrame(
    X_lda,
    index=df.index,
    columns=[f"LD{i+1}" for i in range(X_lda.shape[1])]
)
scores_df.to_csv("./dataOUT/lda_scores.csv")

# coeff/loadings like
coef_df = pd.DataFrame(
    lda.scalings_[:, :X_lda.shape[1]],
    index=X_cols,
    columns=[f"LD{i+1}" for i in range(X_lda.shape[1])]
)
coef_df.to_csv("./dataOUT/lda_coefficients.csv")

# prediction
y_pred = lda.predict(X_std)
cm = confusion_matrix(y, y_pred)
acc = accuracy_score(y, y_pred)

pd.DataFrame(cm).to_csv("./dataOUT/lda_confusion_matrix.csv", index=False)
pd.DataFrame({"accuracy":[acc]}).to_csv("./dataOUT/lda_accuracy.csv", index=False)

# plt
if X_lda.shape[1] == 1:
    plt.figure()
    plt.hist(scores_df["LD1"])
    plt.title("LDA scores (LD1)")
    plt.grid()
    plt.show()
else:
    plt.figure(figsize=(7,7))
    plt.scatter(scores_df["LD1"], scores_df["LD2"], marker="*")
    for i, label in enumerate(df.index):
        plt.text(scores_df.iloc[i,0], scores_df.iloc[i,1], str(label), fontsize=8)
    plt.axhline(0, linestyle="--")
    plt.axvline(0, linestyle="--")
    plt.xlabel("LD1")
    plt.ylabel("LD2")
    plt.title("LDA Score Plot (LD1 vs LD2)")
    plt.grid()
    plt.show()
