"""
STAGE 3 - Load saved models, create new data, pass it to the models, generate results.
Run:  python inference.py
The same predict_all() function is used by the Flask frontend (app.py).
"""
import joblib
import numpy as np
import pandas as pd

_meta = joblib.load("models/scaler.joblib")
_models = joblib.load("models/cluster_models.joblib")
SCALER, COLS = _meta["scaler"], _meta["columns"]


def predict_all(new_df: pd.DataFrame) -> pd.DataFrame:
    """new_df must contain the same feature columns used in training (e.g. age, salary)."""
    X = SCALER.transform(new_df[COLS])
    out = new_df[COLS].copy()
    out["KMeans (K=2)"] = _models["kmeans2"].predict(X)
    out["KMeans (K=3)"] = _models["kmeans3"].predict(X)
    # Hierarchical has no predict(): assign to nearest cluster centroid
    d = np.linalg.norm(X[:, None, :] - _models["hier_centroids"][None], axis=2)
    out["Hierarchical (3)"] = d.argmin(axis=1)
    # DBSCAN has no predict(): nearest core point within eps, otherwise noise (-1)
    core, core_lab, eps = _models["dbscan_core"], _models["dbscan_core_labels"], _models["dbscan_eps"]
    dc = np.linalg.norm(X[:, None, :] - core[None], axis=2)
    out["DBSCAN"] = np.where(dc.min(axis=1) <= eps, core_lab[dc.argmin(axis=1)], -1)
    out["GMM (3)"] = _models["gmm"].predict(X)
    return out


if __name__ == "__main__":
    new_people = pd.DataFrame({"age": [24, 35, 55, 40, 30],
                               "salary": [45000, 140000, 300000, 450000, 90000]})
    result = predict_all(new_people)
    print(result.to_string(index=False))
    result.to_csv("outputs/new_predictions.csv", index=False)
    print("\n(-1 in DBSCAN = noise / outlier)  Saved -> outputs/new_predictions.csv")