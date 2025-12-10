import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap
import pandas as pd


# ============================================================
# 1. Полиномиальные признаки
# ============================================================
def add_polynomial_features(X: pd.DataFrame, degree=2):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_data = poly.fit_transform(X)
    poly_cols = poly.get_feature_names_out(X.columns)

    df_poly = pd.DataFrame(poly_data, columns=poly_cols, index=X.index)

    # Удаляем оригинальные столбцы, чтобы не дублировать
    df_poly = df_poly.drop(columns=X.columns, errors="ignore")

    return pd.concat([X, df_poly], axis=1)



# ============================================================
# 2. Статистические агрегаты
# ============================================================
def add_stat_features(X: pd.DataFrame):
    X_new = X.copy()
    X_new["feat_sum"]  = X.sum(axis=1)
    X_new["feat_mean"] = X.mean(axis=1)
    X_new["feat_std"]  = X.std(axis=1)
    X_new["feat_min"]  = X.min(axis=1)
    X_new["feat_max"]  = X.max(axis=1)
    X_new["feat_cv"]   = X.std(axis=1) / (X.mean(axis=1) + 1e-9)
    return X_new



# ============================================================
# 3. Отношения признаков (ratios)
# ============================================================
def add_ratio_features(X: pd.DataFrame):
    X_new = X.copy()
    cols = X.columns

    # Все попарные отношения
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            X_new[f"{c1}_to_{c2}"] = X[c1] / (X[c2] + 1e-9)
            X_new[f"{c2}_to_{c1}"] = X[c2] / (X[c1] + 1e-9)

    return X_new



# ============================================================
# 4. Логарифмические признаки
# ============================================================
def add_log_features(X: pd.DataFrame):
    X_new = X.copy()
    for col in X.columns:
        X_new["log_" + col] = np.log1p(X[col])
    return X_new



# ============================================================
# 5. PCA признаки
# ============================================================
def add_pca_features(X: pd.DataFrame, n_components=3):
    pca = PCA(n_components=n_components)
    pca_res = pca.fit_transform(X)
    df_pca = pd.DataFrame(
        pca_res,
        columns=[f"PCA{i+1}" for i in range(n_components)],
        index=X.index,
    )
    return pd.concat([X, df_pca], axis=1)



# ============================================================
# 6. UMAP признаки
# ============================================================
def add_umap_features(X: pd.DataFrame, n_components=2, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        # random_state=42,
    )
    emb = reducer.fit_transform(X)

    df_umap = pd.DataFrame(
        emb,
        columns=[f"UMAP{i+1}" for i in range(n_components)],
        index=X.index,
    )
    return pd.concat([X, df_umap], axis=1)



# ============================================================
# 7. Кластеризационные признаки
# ============================================================
def add_cluster_features(X: pd.DataFrame, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    X_new = X.copy()
    X_new["cluster"] = labels

    # Расстояния до центров — очень сильные признаки
    dists = kmeans.transform(X)
    for i in range(n_clusters):
        X_new[f"dist_to_cluster_{i}"] = dists[:, i]

    return X_new
