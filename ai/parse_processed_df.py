# feature_selection_utils.py
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.pipeline import make_pipeline
from sklearn.utils.multiclass import type_of_target


# ---------- utilities ----------
def _safe_copy_and_clean(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    # replace infs and large values
    X = X.replace([np.inf, -np.inf], np.nan)
    # fillna with column median (safe default)
    X = X.fillna(X.median())
    return X


# ---------- 0. quick summary ----------
def summarize_features(X: pd.DataFrame) -> pd.DataFrame:
    Xc = _safe_copy_and_clean(X)
    desc = pd.DataFrame({
        'dtype': Xc.dtypes,
        'n_unique': Xc.nunique(),
        'n_null': Xc.isnull().sum(),
        'mean': Xc.mean(),
        'std': Xc.std(),
        'min': Xc.min(),
        'max': Xc.max(),
        'skew': Xc.skew(),
    })
    desc['abs_mean_over_std'] = (desc['mean'].abs() / (desc['std'] + 1e-12))
    return desc.sort_values('std')


# ---------- 1. remove zero/low variance ----------
def remove_low_variance(X: pd.DataFrame, threshold: float = 1e-8) -> pd.DataFrame:
    Xc = _safe_copy_and_clean(X)
    sel = VarianceThreshold(threshold=threshold)
    sel.fit(Xc)
    cols = Xc.columns[sel.get_support()]
    return Xc[cols]


# ---------- 2. remove duplicate / near-duplicate columns ----------
def remove_duplicate_columns(X: pd.DataFrame, tol: float = 1e-12) -> pd.DataFrame:
    Xc = _safe_copy_and_clean(X)
    # drop exactly duplicated columns first
    Xc = Xc.loc[:, ~Xc.T.duplicated()]
    # drop columns that are linear duplicates (correlation ~ 1)
    corr = Xc.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > (1 - 1e-12))]
    return Xc.drop(columns=to_drop)


# ---------- 3. remove highly correlated features ----------
def remove_highly_correlated(X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    Xc = _safe_copy_and_clean(X)
    corr = Xc.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return Xc.drop(columns=to_drop), to_drop


# ---------- 4. unsupervised feature clustering — выбираем репрезентанты ----------
def feature_cluster_select(X: pd.DataFrame, n_clusters: int = 20, strategy: str = 'variance') -> Tuple[List[str], pd.DataFrame]:
    """
    Clusters features into n_clusters using FeatureAgglomeration.
    For each cluster, selects one representative feature:
      - 'variance' -> feature with largest variance in the cluster
      - 'corr_to_component' -> feature with highest absolute corr to cluster component
    Returns (list_of_selected_feature_names, reduced_df).
    """
    Xc = _safe_copy_and_clean(X)
    # standardize features (important for clustering)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xc)

    agg = FeatureAgglomeration(n_clusters=max(1, min(n_clusters, Xc.shape[1])), metric='euclidean', linkage='ward')
    agg.fit(Xs)
    labels = getattr(agg, "labels_", None)
    if labels is None:
        # fallback: treat as 1 cluster
        labels = np.zeros(Xc.shape[1], dtype=int)

    # get transformed components
    components = agg.transform(Xs)  # shape (n_samples, n_clusters)

    selected = []
    for cl in range(components.shape[1]):
        idxs = np.where(labels == cl)[0]
        feats = Xc.columns[idxs].tolist()
        if strategy == 'variance':
            # pick feature with maximum variance
            variances = Xc[feats].var(axis=0)
            pick = variances.idxmax()
        else:
            # correlation to component
            comp = components[:, cl]
            corrs = {f: abs(np.corrcoef(Xc[f].values, comp)[0, 1]) for f in feats}
            pick = max(corrs, key=corrs.get)
        selected.append(pick)

    selected = list(dict.fromkeys(selected))  # keep order, unique
    return selected, Xc[selected]


# ---------- 5. unsupervised PCA-based selection ----------
def pca_feature_loadings_select(X: pd.DataFrame, n_components: int = 5, top_k_per_comp: int = 5) -> Tuple[List[str], pd.DataFrame]:
    """
    Fit PCA, for each of top n_components pick top_k_per_comp features by absolute loading.
    """
    Xc = _safe_copy_and_clean(X)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xc)
    pca = PCA(n_components=min(n_components, Xs.shape[1]))
    pca.fit(Xs)
    loadings = np.abs(pca.components_)  # shape (n_components, n_features)
    selected = set()
    for i in range(loadings.shape[0]):
        idxs = np.argsort(loadings[i])[::-1][:top_k_per_comp]
        for j in idxs:
            selected.add(Xc.columns[j])
    selected = list(selected)
    return selected, Xc[selected]


# ---------- 6. supervised: mutual information ----------
def select_k_best_mutual_info(X: pd.DataFrame, y: pd.Series, k: int = 30) -> Tuple[List[str], pd.DataFrame]:
    Xc = _safe_copy_and_clean(X)
    # detect type
    t = type_of_target(y)
    if t in ('continuous', 'continuous-multioutput'):
        mi = mutual_info_regression(Xc, y)
    else:
        mi = mutual_info_classif(Xc, y)
    mi = np.array(mi)
    order = np.argsort(mi)[::-1][:min(k, Xc.shape[1])]
    selected = Xc.columns[order].tolist()
    return selected, Xc[selected]


# ---------- 7. supervised: model-based (RandomForest / SelectFromModel) ----------
def select_via_model_importance(X: pd.DataFrame, y: pd.Series, n_features_to_select: int = 30, problem: str = 'auto', random_state: int = 42) -> Tuple[List[str], pd.DataFrame, np.ndarray]:
    """
    Trains a RandomForest and uses feature importances to select top features.
    Returns (selected_feature_names, reduced_df, importances_array_sorted_by_feature_names).
    """
    Xc = _safe_copy_and_clean(X)
    # scale not strictly necessary for tree models, but safe
    # detect problem type
    if problem == 'auto':
        t = type_of_target(y)
        if t in ('continuous', 'continuous-multioutput'):
            problem = 'regression'
        else:
            problem = 'classification'

    if problem == 'regression':
        model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=random_state)
    else:
        model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state)

    model.fit(Xc, y)
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:min(n_features_to_select, Xc.shape[1])]
    selected = Xc.columns[order].tolist()
    importances_by_feature = pd.Series(importances, index=Xc.columns).sort_values(ascending=False)
    return selected, Xc[selected], importances_by_feature


# ---------- 8. L1-based sparse selection (Logistic/Lasso) ----------
def select_via_l1(X: pd.DataFrame, y: pd.Series, n_features_to_select: Optional[int] = None, problem: str = 'auto', random_state: int = 42) -> Tuple[List[str], pd.DataFrame]:
    Xc = _safe_copy_and_clean(X)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xc)

    if problem == 'auto':
        t = type_of_target(y)
        if t in ('continuous', 'continuous-multioutput'):
            problem = 'regression'
        else:
            problem = 'classification'

    if problem == 'classification':
        model = LogisticRegression(penalty='l1', solver='saga', max_iter=2000, random_state=random_state)
        # you may want CV; here simple
        model.fit(Xs, y)
        coefs = np.abs(model.coef_).sum(axis=0)
    else:
        model = LassoCV(cv=5, random_state=random_state, max_iter=20000)
        model.fit(Xs, y)
        coefs = np.abs(model.coef_)

    feat_series = pd.Series(coefs, index=Xc.columns)
    feat_series = feat_series.sort_values(ascending=False)
    if n_features_to_select is None:
        # choose non-zero features
        selected = feat_series[feat_series > 1e-8].index.tolist()
    else:
        selected = feat_series.index[:n_features_to_select].tolist()
    return selected, Xc[selected]


# ---------- 9. combined pipeline for convenience ----------
def auto_feature_selector(X: pd.DataFrame, y: Optional[pd.Series] = None,
                          unsupervised_keep: int = 50,
                          supervised_keep: int = 40,
                          intermediate_corr_threshold: float = 0.98) -> Tuple[List[str], pd.DataFrame, dict]:
    """
    Full pipeline:
      - cleaning
      - remove low-variance + duplicates
      - remove extremely correlated (threshold)
      - if y is None: unsupervised selection (feature clustering + PCA)
      - if y provided: combine mutual info + model importance + L1
    Returns (selected_features, X_reduced, diagnostics dict)
    """
    Xc = _safe_copy_and_clean(X)

    # 1. low-variance
    Xv = remove_low_variance(Xc, threshold=1e-12)

    # 2. remove duplicate columns
    Xd = remove_duplicate_columns(Xv)

    # 3. remove extremely correlated features
    Xr, dropped_corr = remove_highly_correlated(Xd, threshold=intermediate_corr_threshold)

    diagnostics = {
        'n_initial': X.shape[1],
        'n_after_var': Xv.shape[1],
        'n_after_dup': Xd.shape[1],
        'n_after_corr': Xr.shape[1],
        'dropped_corr': dropped_corr
    }

    if y is None:
        # unsupervised strategy: feature clustering + PCA loadings
        sel1, df1 = feature_cluster_select(Xr, n_clusters=min(unsupervised_keep, Xr.shape[1]), strategy='corr_to_component')
        sel2, df2 = pca_feature_loadings_select(Xr, n_components=8, top_k_per_comp=3)
        # union (preserve order)
        selected = list(dict.fromkeys(sel1 + sel2))
        selected = selected[:unsupervised_keep]
        return selected, Xr[selected], diagnostics
    else:
        # supervised: mutual info + model importances + l1
        sel_mi, _ = select_k_best_mutual_info(Xr, y, k=supervised_keep)
        sel_rf, _, importances = select_via_model_importance(Xr, y, n_features_to_select=supervised_keep)
        sel_l1, _ = select_via_l1(Xr, y, n_features_to_select=supervised_keep)
        # create ranked union: weight model importance highest, then MI, then L1
        combined = []
        for s in [list(importances.index), sel_mi, sel_l1]:
            for f in s:
                if f not in combined and f in Xr.columns:
                    combined.append(f)
        selected = combined[:supervised_keep]
        diagnostics.update({'mi_top': sel_mi[:10], 'rf_top': list(importances.index[:10]), 'l1_top': sel_l1[:10]})
        return selected, Xr[selected], diagnostics
