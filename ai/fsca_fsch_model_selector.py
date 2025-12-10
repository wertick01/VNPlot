import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KDTree
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm

class FlowDBSCANSelector:
    """
    Класс для обучения DBSCAN на паре признаков (x,y), предсказания и пакетной
    обработки файлов с визуализацией. Дополнительно: удобные методы для получения
    DataFrame с добавленными метками и сохранения только выбранных (predicted) строк.
    """
    def __init__(self,
                 eps: float = 0.15,
                 min_samples: int = 50,
                 log: bool = True,
                 lower_p: float = 1,
                 upper_p: float = 99,
                 eps_shift: float = 1e-9):
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.log = bool(log)
        self.lower_p = float(lower_p)
        self.upper_p = float(upper_p)
        self.eps_shift = float(eps_shift)

        self.db = None
        self.scaler = None
        self.tree = None
        self.core_points = None
        self.core_labels = None
        self.meta = None
        self._fitted = False

    # ---------------- preprocessing ----------------
    def _preprocess(self, x, y, user_thresholds=None):
        x = np.asarray(x)
        y = np.asarray(y)
        orig_len = len(x)

        mask_not_nan = (~np.isnan(x)) & (~np.isnan(y))
        x0 = x[mask_not_nan].astype(float)
        y0 = y[mask_not_nan].astype(float)

        if user_thresholds is None:
            if x0.size == 0 or y0.size == 0:
                lo_x = hi_x = lo_y = hi_y = 0.0
            else:
                lo_x, hi_x = np.percentile(x0, [self.lower_p, self.upper_p])
                lo_y, hi_y = np.percentile(y0, [self.lower_p, self.upper_p])
        else:
            lo_x = user_thresholds['lo_x']; hi_x = user_thresholds['hi_x']
            lo_y = user_thresholds['lo_y']; hi_y = user_thresholds['hi_y']

        if x0.size == 0:
            x_clip = np.empty(0); y_clip = np.empty(0)
        else:
            x_clip = np.clip(x0, lo_x, hi_x)
            y_clip = np.clip(y0, lo_y, hi_y)

        if self.log:
            def safe_log10(arr):
                arr_safe = arr + self.eps_shift
                arr_safe[arr_safe <= 0] = np.nan
                return np.log10(arr_safe)
            x_proc = safe_log10(x_clip)
            y_proc = safe_log10(y_clip)
        else:
            x_proc = x_clip
            y_proc = y_clip

        if x_proc.size == 0:
            mask_finite = np.array([], dtype=bool)
        else:
            mask_finite = np.isfinite(x_proc) & np.isfinite(y_proc)
            x_proc = x_proc[mask_finite]
            y_proc = y_proc[mask_finite]

        XYs = np.vstack([x_proc, y_proc]).T if x_proc.size > 0 else np.empty((0,2))

        mask_kept = np.zeros(orig_len, dtype=bool)
        if mask_not_nan.any():
            idx_not_nan = np.nonzero(mask_not_nan)[0]
            if mask_finite.size > 0:
                idx_kept = idx_not_nan[mask_finite]
                mask_kept[idx_kept] = True

        meta = {'lo_x': float(lo_x), 'hi_x': float(hi_x),
                'lo_y': float(lo_y), 'hi_y': float(hi_y),
                'log': bool(self.log)}

        return XYs, x_proc, y_proc, meta, mask_kept

    # ---------------- fit / train ----------------
    def fit(self, x, y, sample_size=None, random_state=0):
        t0 = time.time()
        x = np.asarray(x); y = np.asarray(y)
        n = len(x)

        if (sample_size is not None) and (sample_size < n):
            rng = np.random.RandomState(random_state)
            idx = rng.choice(n, size=sample_size, replace=False)
            x_train = x[idx]; y_train = y[idx]
        else:
            x_train = x; y_train = y

        XY_train, x_trim, y_trim, meta, mask_kept_train = self._preprocess(
            x_train, y_train, user_thresholds=None
        )
        self.meta = meta

        self.scaler = RobustScaler()
        XYs_scaled = self.scaler.fit_transform(XY_train) if XY_train.size > 0 else np.empty((0,2))

        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
        if XYs_scaled.size > 0:
            db.fit(XYs_scaled)
            labels = db.labels_
        else:
            labels = np.array([], dtype=int)
        t1 = time.time()

        core_idx = getattr(db, "core_sample_indices_", np.array([], dtype=int))
        if len(core_idx) == 0:
            core_points = np.empty((0,2)); core_labels = np.array([], dtype=int); tree = None
        else:
            core_points = XYs_scaled[core_idx]
            core_labels = labels[core_idx]
            tree = KDTree(core_points)

        t2 = time.time()

        self.db = db
        self.tree = tree
        self.core_points = core_points
        self.core_labels = core_labels
        self._fitted = True

        self.train_info = {
            'orig_train_size': len(x_train),
            'used_after_preprocess': XY_train.shape[0],
            'train_time': t1 - t0,
            'postproc_time': t2 - t1,
            'n_cores': len(core_idx)
        }
        print(f"Train: preprocess+fit DBSCAN: {self.train_info['train_time']:.2f}s, "
              f"tree build: {self.train_info['postproc_time']:.2f}s, cores: {self.train_info['n_cores']}")
        return self

    def fit_from_df(self, df: pd.DataFrame, x_col="FSC-A", y_col="SSC-A",
                    sample_size=None, random_state=0):
        x = df[x_col].to_numpy(); y = df[y_col].to_numpy()
        return self.fit(x, y, sample_size=sample_size, random_state=random_state)

    # ---------------- predict ----------------
    def predict(self, x_new, y_new, assign_method='nearest'):
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call fit(...) first.")

        x_new = np.asarray(x_new); y_new = np.asarray(y_new)
        n_all = len(x_new)

        user_th = {'lo_x': self.meta['lo_x'], 'hi_x': self.meta['hi_x'],
                   'lo_y': self.meta['lo_y'], 'hi_y': self.meta['hi_y']}
        XY_new_raw, x_clip, y_clip, _, mask_kept = self._preprocess(x_new, y_new, user_thresholds=user_th)

        labels_full = np.full(n_all, -1, dtype=int)
        if XY_new_raw.size == 0:
            return labels_full

        XY_new_scaled = self.scaler.transform(XY_new_raw)

        if (self.tree is None) or (self.core_points is None) or (len(self.core_points) == 0):
            return labels_full

        dist, idx = self.tree.query(XY_new_scaled, k=1)
        dist = dist.ravel(); idx = idx.ravel()
        labels_assigned = np.full(len(XY_new_scaled), -1, dtype=int)
        within = dist <= self.eps
        labels_assigned[within] = self.core_labels[idx[within]]

        labels_full[mask_kept] = labels_assigned
        return labels_full

    # ---------------- DataFrame helpers ----------------
    def predict_df(self, df, x_col, y_col, labels_col='dbscan_label', inplace=False):
        """
        Return DataFrame with added column labels_col (copy by default).
        """
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call fit(...) first.")
        if inplace:
            df_out = df
        else:
            df_out = df.copy()
        labels = self.predict(df_out[x_col].to_numpy(), df_out[y_col].to_numpy())
        df_out[labels_col] = labels
        df_out = df_out[df_out[labels_col]==0]
        return df_out

    def get_selected_df(self, df: pd.DataFrame, x_col="FSC-A", y_col="SSC-A", labels_col='dbscan_label'):
        """
        Returns a new DataFrame containing only rows with labels != -1.
        """
        df_with = self.predict_df(df, x_col=x_col, y_col=y_col, labels_col=labels_col, inplace=False)
        df_sel = df_with[df_with[labels_col] != -1].copy()
        return df_sel

    def save_selected_df(self, path, df=None, df_original=None, x_col="FSC-A", y_col="SSC-A", labels_col='dbscan_label'):
        """
        Save selected dataframe to `path`. Either provide `df` (already filtered selected rows),
        or provide `df_original` (full DataFrame) and method will compute selected rows and save them.
        Supported formats: .parquet, .csv (determined by path suffix).
        """
        if df is None and df_original is None:
            raise ValueError("Provide either df (filtered) or df_original (full) to compute selection).")

        if df is None:
            df_to_save = self.get_selected_df(df_original, x_col=x_col, y_col=y_col, labels_col=labels_col)
        else:
            df_to_save = df.copy()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suf = path.suffix.lower()
        if suf in ['.parquet', '.pq']:
            df_to_save.to_parquet(path, index=False)
        elif suf in ['.csv', '.txt']:
            df_to_save.to_csv(path, index=False)
        else:
            raise ValueError("Unsupported save format. Use .parquet or .csv")

        return path

    # ---------------- save / load model ----------------
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)

    # ---------------- batch processing & visualization ----------------
    def process_and_visualize_files(self,
                                    file_paths,
                                    x_col="FSC-A",
                                    y_col="SSC-A",
                                    read_chunksize=None,
                                    sample_frac_for_plot=0.05,
                                    sample_cap=200_000,
                                    out_dir="results_vis",
                                    rng_seed=0):
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call fit(...) first.")

        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        rows = []; rng = np.random.RandomState(rng_seed)

        for fp in file_paths:
            fp = Path(fp)
            print(f"\n--- Processing file: {fp} ---")
            t0_file = time.perf_counter()

            if fp.suffix.lower() in [".parquet", ".pq"]:
                df = pd.read_parquet(fp); read_time = 0.0; print(f"Read parquet: {len(df):,} rows")
            elif fp.suffix.lower() in [".csv", ".txt"]:
                if read_chunksize is None:
                    t0r = time.perf_counter(); df = pd.read_csv(fp); read_time = time.perf_counter() - t0r
                    print(f"Read csv full: {len(df):,} rows, {read_time:.2f}s")
                else:
                    t0r = time.perf_counter(); chunks = []
                    for chunk in pd.read_csv(fp, usecols=[x_col, y_col], chunksize=read_chunksize):
                        chunks.append(chunk)
                    df = pd.concat(chunks, ignore_index=True); read_time = time.perf_counter() - t0r
                    print(f"Read csv by chunks -> total rows: {len(df):,}, read time {read_time:.2f}s")
            else:
                raise ValueError(f"Unsupported file extension: {fp.suffix}")

            x_all = df[x_col].to_numpy(); y_all = df[y_col].to_numpy()
            t0_pred = time.perf_counter(); labels = self.predict(x_all, y_all); pred_time = time.perf_counter() - t0_pred

            mask_selected = labels != -1; total = len(labels); selected = int(mask_selected.sum())
            frac = selected / total if total > 0 else 0.0

            def safe_stats(arr, mask=None):
                if mask is None:
                    a = arr[~np.isnan(arr)]
                else:
                    a = arr[mask & (~np.isnan(arr))]
                if len(a) == 0:
                    return {"count":0, "mean":np.nan, "med":np.nan, "std":np.nan}
                return {"count":len(a), "mean":float(np.nanmean(a)), "med":float(np.nanmedian(a)), "std":float(np.nanstd(a))}

            stats_all_x = safe_stats(x_all, None); stats_sel_x = safe_stats(x_all, mask_selected)
            stats_all_y = safe_stats(y_all, None); stats_sel_y = safe_stats(y_all, mask_selected)

            rows.append({
                "file": str(fp), "rows": total, "selected": selected, "fraction": frac,
                "read_time_s": read_time, "predict_time_s": pred_time,
                "mean_x_all": stats_all_x["mean"], "mean_x_sel": stats_sel_x["mean"],
                "med_x_all": stats_all_x["med"], "med_x_sel": stats_sel_x["med"],
                "mean_y_all": stats_all_y["mean"], "mean_y_sel": stats_sel_y["mean"],
                "med_y_all": stats_all_y["med"], "med_y_sel": stats_sel_y["med"],
            })

            sample_n = min(int(total * sample_frac_for_plot), sample_cap)
            sample_n = max(sample_n, 2000) if total >= 2000 else total
            idx_plot = rng.choice(total, size=sample_n, replace=False) if sample_n < total else np.arange(total)

            XY_plot_x = x_all[idx_plot]; XY_plot_y = y_all[idx_plot]; labels_plot = labels[idx_plot]; mask_plot_sel = labels_plot != -1

            fig = plt.figure(figsize=(14,6))
            gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1], hspace=0.3, wspace=0.3)
            ax_scatter = fig.add_subplot(gs[:, 0]); ax_histx = fig.add_subplot(gs[0, 1]); ax_histy = fig.add_subplot(gs[1, 1])

            ax_scatter.scatter(XY_plot_x, XY_plot_y, s=1, alpha=0.15, color="gray", label="all (sampled)")
            ax_scatter.scatter(XY_plot_x[mask_plot_sel], XY_plot_y[mask_plot_sel], s=2, alpha=0.7, color="tab:blue", label="selected")
            ax_scatter.set_xlabel(x_col); ax_scatter.set_ylabel(y_col)
            ax_scatter.set_title(f"{fp.name} — selected {selected}/{total} ({frac:.3%})")
            ax_scatter.legend(markerscale=4, fontsize=9)

            bins_x = 100; bins_y = 100
            ax_histx.hist(XY_plot_x, bins=bins_x, alpha=0.25, density=True, label="all")
            ax_histx.hist(XY_plot_x[mask_plot_sel], bins=bins_x, alpha=0.6, density=True, label="selected")
            ax_histx.set_title(f"{x_col} marginal"); ax_histx.legend(fontsize=8)

            ax_histy.hist(XY_plot_y, bins=bins_y, alpha=0.25, density=True, label="all")
            ax_histy.hist(XY_plot_y[mask_plot_sel], bins=bins_y, alpha=0.6, density=True, label="selected")
            ax_histy.set_title(f"{y_col} marginal"); ax_histy.legend(fontsize=8)

            try:
                xmin, xmax = np.percentile(XY_plot_x, [1, 99]); ymin, ymax = np.percentile(XY_plot_y, [1, 99])
                ax_scatter.set_xlim(xmin, xmax); ax_scatter.set_ylim(ymin, ymax)
            except Exception:
                pass

            stats_txt = (
                f"rows: {total:,}\nselected: {selected:,} ({frac:.3%})\n\n"
                f"mean x all/sel: {stats_all_x['mean']:.2f}/{stats_sel_x['mean']:.2f}\n"
                f"med x all/sel: {stats_all_x['med']:.2f}/{stats_sel_x['med']:.2f}\n\n"
                f"mean y all/sel: {stats_all_y['mean']:.2f}/{stats_sel_y['mean']:.2f}\n"
                f"med y all/sel: {stats_all_y['med']:.2f}/{stats_sel_y['med']:.2f}\n"
            )
            fig.text(0.02, 0.98, stats_txt, va="top", ha="left", fontsize=9, family="monospace",
                     bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

            out_fig = out_dir / f"{fp.stem}_selection.png"
            fig.savefig(out_fig, dpi=180, bbox_inches="tight"); plt.close(fig)

            t_file_done = time.perf_counter()
            print(f"Predict time: {pred_time:.2f}s, plotting & saving: {t_file_done - t0_file - pred_time:.2f}s, total: {t_file_done - t0_file:.2f}s")

        summary = pd.DataFrame(rows)
        summary_path = out_dir / "summary_selection.csv"
        summary.to_csv(summary_path, index=False)
        print("\nAll done. Summary saved to:", summary_path)
        return summary

class DBSCANModelManager:
    def __init__(self):
        self.models = {}

    def add_model(self, name, model):
        self.models[name] = model

    def predict_array(self, model_name, x, y):
        model = self.models[model_name]
        return model.predict(x, y)

    def predict_df(self, model_name, df, x_col, y_col):
        x = df[x_col].values
        y = df[y_col].values
        labels = self.predict_array(model_name, x, y)
        df_filtered = df[labels == 0].copy()
        df_filtered['cluster_label'] = labels[labels == 0]
        return df_filtered, labels

    def predict_files_with_fallback(
        self,
        file_paths,
        model_names,
        x_col="FSC-H",
        y_col="FSC-A",
        out_dir="filtered_results",
        min_rows=10_000,
        min_frac=0.75
    ):
        """
        model_names: список из трёх моделей по приоритету [primary, backup1, backup2]
        min_frac: минимальная доля выбранных точек, ниже которой переключаемся на резервную модель
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_rows = []

        for fp in tqdm(file_paths, desc="Processing files", unit="file"):
            fp = Path(fp)
            try:
                df = pd.read_parquet(fp)
            except Exception as e:
                print(f"Failed to read {fp}: {e}")
                continue

            total_rows = len(df)
            if total_rows < min_rows:
                continue

            df_filtered = None
            frac_selected = 0
            used_model = None

            for model_name in model_names:
                df_filtered_candidate, labels = self.predict_df(model_name, df, x_col, y_col)
                frac_selected = len(df_filtered_candidate) / total_rows
                if frac_selected >= min_frac:
                    df_filtered = df_filtered_candidate
                    used_model = model_name
                    break

            if df_filtered is None:
                # fallback на последнюю модель
                df_filtered, labels = self.predict_df(model_names[-1], df, x_col, y_col)
                used_model = model_names[-1]
                frac_selected = len(df_filtered) / total_rows

            if len(df_filtered) < min_rows:
                continue

            out_fp = out_dir / f"{fp.stem}_2.parquet"
            df_filtered.to_parquet(out_fp)

            summary_rows.append({
                "file": str(fp),
                "total_rows": total_rows,
                "selected_rows": len(df_filtered),
                "frac_selected": frac_selected,
                "used_model": used_model
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_path = out_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")
        return summary_df