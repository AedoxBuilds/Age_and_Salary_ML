"""
STAGE 2 - Load preprocessed data, split, model, train, evaluate, save models.

Clustering is unsupervised, so the "split" here separates the feature matrix
from the display columns (original age/salary used for plots and reports).
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score

CLEAN_PATH = "data/people_preprocessed.csv"
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ---------------- load + split ----------------
df = pd.read_csv(CLEAN_PATH)
meta = joblib.load("models/scaler.joblib")
cols = meta["columns"]
X = df[[f"scaled_{c}" for c in cols]].values          # features for the models
show = df[cols]                                       # original units for plots
xc, yc = cols[0], cols[1]

# ---------------- models ----------------
# Task 2 & 3: K-Means K=2 and K=3
km2 = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
km3 = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
# Task 4: Hierarchical (Ward linkage)
hc = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(X)
# Task 5: DBSCAN (eps chosen for 20 points in standardised space)
db = DBSCAN(eps=0.6, min_samples=3).fit(X)
# Task 6: Gaussian Mixture, 3 components
gmm = GaussianMixture(n_components=3, covariance_type="full", n_init=10, random_state=42).fit(X)

labels = {
    "KMeans (K=2)": km2.labels_,
    "KMeans (K=3)": km3.labels_,
    "Hierarchical (3)": hc.labels_,
    "DBSCAN": db.labels_,
    "GMM (3)": gmm.predict(X),
}
for name, lab in labels.items():
    df[name] = lab

# ---------------- evaluate ----------------
def sil(lab):
    mask = lab != -1                                   # ignore DBSCAN noise
    n = len(set(lab[mask]))
    return silhouette_score(X[mask], lab[mask]) if 2 <= n < mask.sum() else np.nan

rows = []
for name, lab in labels.items():
    n_clusters = len(set(lab)) - (1 if -1 in lab else 0)
    rows.append({"Algorithm": name, "Clusters": n_clusters,
                 "Noise points": int((lab == -1).sum()),
                 "Silhouette": round(sil(lab), 3),
                 "Sizes": dict(sorted(pd.Series(lab).value_counts().items()))})
summary = pd.DataFrame(rows)

print("\n=== Task 3: K=2 vs K=3 ===")
print(summary.iloc[:2].to_string(index=False))
print("\n=== Task 5: DBSCAN noise points ===")
print(df.loc[df["DBSCAN"] == -1, ["person_id", xc, yc]].to_string(index=False)
      if "person_id" in df else df.loc[df["DBSCAN"] == -1, [xc, yc]])
print("\n=== Task 8: comparison of all algorithms ===")
print(summary.to_string(index=False))

names = list(labels)
ari = pd.DataFrame([[adjusted_rand_score(labels[a], labels[b]) for b in names] for a in names],
                   index=names, columns=names).round(2)
print("\nAgreement between algorithms (Adjusted Rand Index, 1 = identical):")
print(ari.to_string())

# ---------------- visualise (Task 7) ----------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, (name, lab) in zip(axes.ravel(), labels.items()):
    for k in sorted(set(lab)):
        m = lab == k
        if k == -1:
            ax.scatter(show[xc][m], show[yc][m], c="black", marker="x", s=90, label="Noise")
        else:
            ax.scatter(show[xc][m], show[yc][m], s=70, edgecolor="k", label=f"Cluster {k}")
    ax.set_title(name); ax.set_xlabel(xc.title()); ax.set_ylabel(yc.title()); ax.legend(fontsize=8)
    ax.grid(alpha=.3)
# last panel: dendrogram
Z = linkage(X, method="ward")
dendrogram(Z, ax=axes[1, 2], labels=(df["person_id"].values if "person_id" in df else None))
axes[1, 2].set_title("Dendrogram (Ward)")
plt.tight_layout()
plt.savefig("outputs/clusters_comparison.png", dpi=150)

# ---------------- save ----------------
df.to_csv("outputs/clustering_results.csv", index=False)
summary.to_csv("outputs/summary.csv", index=False)

# centroids / reference points so non-predicting algorithms can label new data
hc_centroids = np.array([X[hc.labels_ == k].mean(axis=0) for k in sorted(set(hc.labels_))])
joblib.dump({
    "kmeans2": km2, "kmeans3": km3, "gmm": gmm,
    "hier_centroids": hc_centroids,
    "dbscan_core": db.components_, "dbscan_core_labels": db.labels_[db.core_sample_indices_],
    "dbscan_eps": db.eps,
}, "models/cluster_models.joblib")
print("\nSaved models -> models/cluster_models.joblib, plot -> outputs/clusters_comparison.png")