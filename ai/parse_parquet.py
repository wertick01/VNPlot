import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from feature_selection_utils import *
from parse_processed_df import *
import logging
from sklearn.metrics import roc_auc_score
import json

def process_file(path, l, i):
    try:
        logger.info(f"File {i}/{l} --> {path} processed")
        # file_base = str(path).split("\\")[-1].replace(".parquet", "")
        data = pd.read_parquet(path)
        scaler = MinMaxScaler(feature_range=(1e-5, 1))
        df_scaled = pd.DataFrame(scaler.fit_transform(data[["FSC-H", "SSC-H", "FSC-A", "SSC-A", "FITC-A"]]), columns=data[["FSC-H", "SSC-H", "FSC-A", "SSC-A", "FITC-A"]].columns)
        df_scaled["label"] = data["label"].astype(int)

        # Предположим, что ваш датафрейм называется df
        # Разделяем датафрейм на признаки и целевую переменную
        X = df_scaled.drop('label', axis=1)  # признаки
        y = df_scaled['label']  # целевая переменная

        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X,  y)
        logger.info("Распределение классов до балансировки:", y.value_counts())
        logger.info("Распределение классов после балансировки:", y_balanced.value_counts())

        logger.info(f"File {i}/{l} --> {path} scaled")

        X_balanced["label"] = y_balanced

        return X_balanced

        # logger.info("Start generationg features")
        # df = generate_all_features(df=X_balanced)
        # df["label"] = y_balanced
        # df.to_parquet("G:\\parsed_data\\"+file_base+"_stats.parquet", compression='zstd', compression_level=10)
        # logger.info("Train saved to:", "G:\\parsed_data\\"+file_base+"_stats.parquet")

        # logger.info("Start summarising features")
        # X = df
        # if "label" in X.columns:
        #     X = X.drop(columns=["label"])  # если эти поля — не фичи
        # y = y_balanced   # твоя целевая переменная
        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        # Xv = remove_low_variance(X_train, threshold=1e-12)   # порог можно поднять до 1e-8
        # Xd = remove_duplicate_columns(Xv)
        # print("after var:", Xv.shape, "after dup:", Xd.shape)

        # Xr, dropped = remove_highly_correlated(Xd, threshold=0.95)
        # print("dropped by corr:", dropped)

        # sel_mi, _ = select_k_best_mutual_info(Xr, y_train, k=60)
        # sel_rf, _, importances = select_via_model_importance(Xr, y_train, n_features_to_select=60)
        # sel_l1, _ = select_via_l1(Xr, y_train, n_features_to_select=60)

        # # объединяем с приоритетом RF -> MI -> L1
        # combined = []
        # for s in [list(importances.index), sel_mi, sel_l1]:
        #     for f in s:
        #         if f not in combined and f in Xr.columns:
        #             combined.append(f)
        # selected_supervised = combined[:60]   # оставить top-40 (пример)

        # X_train = X_train[selected_supervised]
        # X_train["label"] = y_train
        # X_train.to_csv("E:\\parquet\\parsed_data\\"+file_base+"_train.parquet")
        # logger.info("Train saved to:", "E:\\parquet\\parsed_data\\"+file_base+"_train.parquet")

        # X_val = X_val[selected_supervised]
        # X_val["label"] = y_val
        # X_val.to_csv("E:\\parquet\\parsed_data\\"+file_base+"_val.parquet")
        # logger.info("Val saved to:", "E:\\parquet\\parsed_data\\"+file_base+"_val.parquet")
        # return

    except:
        logger.info(f"Error with file {i}/{l} --> {file}")
        return

    # clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    # clf.fit(X_train[selected_supervised], y_train)
    # pred = clf.predict_proba(X_val[selected_supervised])[:,1]

    # auc_score = roc_auc_score(y_val, pred)
    # params = clf.get_params().items()
    # params["AUC"] = auc_score
    # params["file"] = file_base
    # return params


def generate_all_features(
        df,
        poly_degree=2,
        pca_components=3,
        umap_components=2,
        n_clusters=5,
        ):
    logger.info(f"Data preview:\n{df.head()}")

    logger.info("Adding polynomial features")
    df = add_polynomial_features(df, degree=poly_degree)

    logger.info("Adding statistical features")
    df = add_stat_features(df)

    logger.info("Adding ratio features")
    df = add_ratio_features(df)

    logger.info("Adding log features")
    df = add_log_features(df)

    logger.info(f"Adding PCA features with {pca_components} components")
    df = add_pca_features(df, n_components=pca_components)

    logger.info(f"Adding UMAP features with {umap_components} components")
    df = add_umap_features(df, n_components=umap_components)

    logger.info(f"Adding clustering features with {n_clusters} clusters")
    df = add_cluster_features(df, n_clusters=n_clusters)

    return df


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger("logger")

folder_path = Path("E:\\parquet")
parquet_files = list(folder_path.glob("*.parquet"))
dfs = []
i = 1
for file in parquet_files:
    df = process_file(file, len(parquet_files), i)
    dfs.append(df)
    i += 1
#     param_list.append(params)

final_df = pd.concat(dfs, ignore_index=True)

# with open("E:\\parquet\\parsed_data\\params.json", "w", encoding="utf-8") as f:
#     json.dump(param_list, f)
final_df.to_parquet("E:\\parquet\\parsed_data\\merged.parquet")
logger.info(f"Saved to E:\\parquet\\parsed_data\\merged.parquet")