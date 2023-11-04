from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# need to be scaled before inputed

exp_var = sum(pca.explained_variance_ratio_ * 100)
print("Variance explained:", exp_var)

import numpy as np

pca = PCA(n_components=None)
X_pca = pca.fit_transform(X_scaled)

exp_var = pca.explained_variance_ratio_ * 100
cum_exp_var = np.cumsum(exp_var)

plt.bar(range(1, 14), exp_var, align="center", label="Individual explained variance")

plt.step(
    range(1, 14),
    cum_exp_var,
    where="mid",
    label="Cumulative explained variance",
    color="red",
)

plt.ylabel("Explained variance percentage")
plt.xlabel("Principal component index")
plt.xticks(ticks=list(range(1, 14)))
plt.legend(loc="best")
plt.tight_layout()

plt.savefig("Barplot_PCA.png")
