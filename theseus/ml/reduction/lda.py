from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis().fit(X, y)  # fitted LDA model
lda.transform(X)

import matplotlib.pyplot as plt

plt.figure(figsize=[7, 5])

plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, s=25, cmap="plasma")
plt.title("LDA for wine data with 2 components")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.savefig("LDA.png")

exp_var = sum(lda.explained_variance_ratio_ * 100)
print("Variance explained:", exp_var)
