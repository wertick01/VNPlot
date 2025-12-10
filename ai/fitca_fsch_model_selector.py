import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.stats import norm
import joblib
from pathlib import Path


class FlowGMMSelector1D:
    """
    Класс для работы с одномерными данными через GMM.
    Фильтрует NaN и Inf, поддерживает безопасное логарифмирование,
    возвращает DataFrame с меткой выбранного компонента.
    """
    def __init__(self, n_components=2, log=True, eps_shift=1e-9):
        self.n_components = n_components
        self.log = log
        self.eps_shift = eps_shift
        self.gmm = None
        self.right_comp = None
        self.meta = None
        self._fitted = False

    def save(self, path):
        """Сохранить объект класса целиком (joblib)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path):
        """Загрузить объект (joblib)."""
        return joblib.load(path)

    def _safe_log1p(self, arr):
        arr_safe = arr + self.eps_shift
        arr_safe[arr_safe <= 0] = np.nan
        return np.log1p(arr_safe)

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        x_valid = x[np.isfinite(x)]
        if x_valid.size == 0:
            raise ValueError("Нет валидных значений для обучения GMM.")
        x_proc = self._safe_log1p(x_valid) if self.log else x_valid
        x_proc = x_proc[np.isfinite(x_proc)].reshape(-1, 1)
        self.gmm = GaussianMixture(n_components=self.n_components, random_state=0)
        self.gmm.fit(x_proc)
        # выбираем «правый» компонент как тот с большим средним
        self.right_comp = np.argmax(self.gmm.means_.ravel())
        self._fitted = True
        self.meta = {"log": self.log, "eps_shift": self.eps_shift}
        return self

    def predict(self, x_new):
        if not self._fitted:
            raise RuntimeError("GMM не обучен. Вызовите fit(x).")
        x_new = np.asarray(x_new, dtype=float)
        labels_full = np.full(len(x_new), -1, dtype=int)

        # маска валидных (не NaN и не Inf) значений
        mask_valid = np.isfinite(x_new)
        if mask_valid.any():
            x_valid = x_new[mask_valid]
            x_proc = self._safe_log1p(x_valid) if self.log else x_valid
            # оставляем только конечные значения после log
            mask_final = np.isfinite(x_proc)
            if mask_final.any():
                x_for_pred = x_proc[mask_final].reshape(-1,1)
                labels_pred = self.gmm.predict(x_for_pred)
                mask_right = (labels_pred == self.right_comp).astype(int)
                # присваиваем обратно только корректные индексы
                valid_idx = np.nonzero(mask_valid)[0][mask_final]
                labels_full[valid_idx] = mask_right
        return labels_full


    def predict_df(self, df, col='FITC-A', label_col='gmm_right', inplace=False):
        if not self._fitted:
            raise RuntimeError("GMM не обучен. Вызовите fit(x).")
        df_out = df if inplace else df.copy()
        labels = self.predict(df_out[col].to_numpy())
        df_out[label_col] = labels
        return df_out

    def get_selected_df(self, df, col='FITC-A', label_col='gmm_right'):
        df_labeled = self.predict_df(df, col=col, label_col=label_col)
        return df_labeled[df_labeled[label_col] == 1].copy()

    def apply_files(self, file_paths, col='FITC-A', out_dir=None):
        """
        Аналогично FlowDBSCANSelector.process_and_visualize_files:
        фильтрует NaN, Inf, добавляет метку компонента GMM, сохраняет по файлам (если out_dir указан).
        """
        import matplotlib.pyplot as plt
        from pathlib import Path
        import os

        out_dir = Path(out_dir) if out_dir is not None else None
        rows = []
        for fp in file_paths:
            fp = Path(fp)
            if fp.suffix.lower() in ['.csv', '.txt']:
                df = pd.read_csv(fp)
            elif fp.suffix.lower() in ['.parquet', '.pq']:
                df = pd.read_parquet(fp)
            else:
                continue

            df_labeled = self.predict_df(df, col=col, label_col='gmm_right')
            mask_selected = df_labeled['gmm_right'] == 1
            total = len(df_labeled)
            selected = mask_selected.sum()
            frac = selected / total if total > 0 else 0

            stats_all = {"mean": float(np.nanmean(df_labeled[col])), "med": float(np.nanmedian(df_labeled[col]))}
            stats_sel = {"mean": float(np.nanmean(df_labeled.loc[mask_selected, col])),
                        "med": float(np.nanmedian(df_labeled.loc[mask_selected, col]))}

            rows.append({
                "file": str(fp), "rows": total, "selected": selected, "fraction": frac,
                "mean_all": stats_all["mean"], "med_all": stats_all["med"],
                "mean_sel": stats_sel["mean"], "med_sel": stats_sel["med"]
            })

            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{fp.stem}_gmm.parquet"
                df_labeled.to_parquet(out_path, index=False)

        summary = pd.DataFrame(rows)
        if out_dir is not None:
            summary.to_csv(out_dir / "summary_gmm.csv", index=False)
        return summary
    
    def process_and_visualize_files(self, in_dir,
                                    x_col="FITC-A", y_col="FSC-H",
                                    out_dir="gmm_results",
                                    sample_frac_for_plot=0.05,
                                    sample_cap=200_000,
                                    biexp=False,
                                    min_rows=0,
                                    model_paths=None,
                                    threshold_pct=35.0,
                                    visualize=True,
                                    recursive=False,
                                    extensions=(".parquet", ".pq", ".csv", ".txt")):
        """
        Обрабатывает все файлы в директории `in_dir`:
        - ищет файлы с расширениями `extensions` (рекурсивно если recursive=True)
        - для каждого файла: пытаетcя применить модели из model_paths (или self)
        - выбирает первую модель, у которой infectivity_percent < threshold_pct
        - сохраняет labeled parquet и (опционально) png-рисунок в out_dir
        - выводит прогресс [i/N] Processing: <filename>
        - возвращает DataFrame summary

        Параметры:
        in_dir: str or Path — директория с файлами
        visualize: bool — сохранять ли PNG (True) или нет (False)
        recursive: bool — использовать рекурсивный поиск (rglob) или нет (glob)
        extensions: tuple — допустимые расширения файлов (по умолчанию parquet/csv/txt)
        """
        from pathlib import Path
        import pandas as pd
        import numpy as np
        import math

        in_dir = Path(in_dir)
        if not in_dir.exists() or not in_dir.is_dir():
            raise ValueError(f"in_dir не найдена или не директория: {in_dir}")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Собираем список файлов
        file_paths = []
        if recursive:
            iterator = in_dir.rglob("*")
        else:
            iterator = in_dir.glob("*")

        exts_lower = tuple(e.lower() for e in extensions)
        for p in iterator:
            if not p.is_file():
                continue
            if p.suffix.lower() in exts_lower:
                file_paths.append(p)

        file_paths = sorted(file_paths)
        total_files = len(file_paths)
        if total_files == 0:
            print(f"No files with extensions {extensions} found in {in_dir}")
            return pd.DataFrame([])

        # Загружаем модели (как в process_files_list)
        models, model_sources = self._load_models_list(model_paths)

        rows = []
        rng = np.random.RandomState(0)

        for idx_file, fp in enumerate(file_paths, start=1):
            print(f"[{idx_file}/{total_files}] Processing: {fp.name}")

            # чтение
            try:
                if fp.suffix.lower() in [".parquet", ".pq"]:
                    df = pd.read_parquet(fp)
                elif fp.suffix.lower() in [".csv", ".txt"]:
                    df = pd.read_csv(fp)
                else:
                    # на всякий случай (хотя отфильтровано выше)
                    print("  Unsupported file:", fp)
                    rows.append({"file": str(fp), "rows": None, "selected": None,
                                "infectivity_percent": None,
                                "model_used": "unsupported", "model_path": None})
                    continue
            except Exception as e:
                print("  Read failed:", fp, e)
                rows.append({"file": str(fp), "rows": None, "selected": None,
                            "infectivity_percent": None,
                            "model_used": "read_failed", "model_path": None})
                continue

            total = len(df)
            if total < min_rows:
                print(f"  Skipped {fp.name}: rows {total} < {min_rows}")
                rows.append({"file": str(fp), "rows": total, "selected": 0,
                            "infectivity_percent": None,
                            "model_used": "skipped_min_rows", "model_path": None})
                continue

            accepted = False
            df_labeled_to_save = None
            used_model_idx = None
            used_model_src = None
            stats_for_save = None

            # пробуем модели по порядку
            for m_idx, model in enumerate(models):
                try:
                    df_labeled = model.predict_df(df, col=x_col, label_col="gmm_right")
                except Exception as e:
                    print(f"   Model {m_idx} predict failed: {e}")
                    continue

                selected = int((df_labeled["gmm_right"] == 1).sum())
                infectivity_percent = 100 * selected / total if total > 0 else 0.0

                print(f"    Model {m_idx}: infectivity_percent = {infectivity_percent:.2f}%")

                if infectivity_percent < threshold_pct:
                    accepted = True
                    used_model_idx = m_idx
                    used_model_src = model_sources[m_idx] if m_idx < len(model_sources) else None
                    df_labeled_to_save = df_labeled

                    # компонентная статистика (если возможно)
                    x_valid = df_labeled[x_col].to_numpy()
                    x_valid = x_valid[np.isfinite(x_valid) & (x_valid > 0)]
                    if x_valid.size > 0 and hasattr(model, "gmm") and model.gmm is not None:
                        comp = model.gmm.predict(np.log1p(x_valid).reshape(-1, 1))
                        uniq, cnt = np.unique(comp, return_counts=True)
                    else:
                        uniq, cnt = np.array([], dtype=int), np.array([], dtype=int)

                    stats = {
                        "file": str(fp),
                        "rows": total,
                        "selected": int(selected),
                        "infectivity_percent": float(infectivity_percent),
                        "model_used": int(m_idx),
                        "model_path": used_model_src
                    }
                    for u, c in zip(uniq, cnt):
                        stats[f"comp_{u}_count"] = int(c)
                        stats[f"comp_{u}_pct"] = float(c / len(comp) * 100.0) if len(comp) > 0 else 0.0

                    stats_for_save = stats
                    break
                else:
                    # пробуем следующую модель
                    continue

            if not accepted:
                print(f"  No model produced infectivity_percent < {threshold_pct}%, skipping file.")
                rows.append({"file": str(fp), "rows": total, "selected": 0,
                            "infectivity_percent": None,
                            "model_used": "skipped", "model_path": None})
                continue

            # сохраняем labeled parquet
            out_parquet = out_dir / f"{fp.stem}_gmm.parquet"
            try:
                df_labeled_to_save.to_parquet(out_parquet, index=False)
            except Exception as e:
                print("  Failed to save parquet:", e)

            # визуализация (опционально)
            if visualize:
                out_png = out_dir / f"{fp.stem}_gmm.png"
                try:
                    # используем gmm выбранной модели (если есть)
                    gmm_to_plot = None
                    if used_model_idx is not None and used_model_idx < len(models):
                        mod = models[used_model_idx]
                        if hasattr(mod, "gmm"):
                            gmm_to_plot = mod.gmm
                    # Если gmm отсутствует, пытаемся взять self.gmm как fallback
                    if gmm_to_plot is None:
                        gmm_to_plot = getattr(self, "gmm", None)

                    if gmm_to_plot is None:
                        print("  No gmm available for plotting, skipping plot.")
                    else:
                        # семплирование для уменьшения размера картинки
                        if sample_frac_for_plot is not None and 0 < sample_frac_for_plot < 1.0:
                            sample_n = min(int(total * sample_frac_for_plot), sample_cap)
                            if sample_n < total:
                                idx_plot = rng.choice(total, size=sample_n, replace=False)
                                df_plot = df_labeled_to_save.iloc[idx_plot]
                            else:
                                df_plot = df_labeled_to_save
                        else:
                            df_plot = df_labeled_to_save

                        # используем метод plot_gmm_full_figure класса (он сохранит картинку)
                        try:
                            self.plot_gmm_full_figure(df_plot, gmm=gmm_to_plot,
                                                    col_x=x_col, col_y=y_col,
                                                    title=f"{fp.name} (model {used_model_idx})",
                                                    biexp=biexp, out_path=out_png)
                        except Exception as e:
                            print("  Plot failed for", fp.name, e)
                except Exception as e:
                    print("  Visualization step failed:", e)

            # добавляем в summary
            rows.append(stats_for_save)

        # общий summary
        summary = pd.DataFrame(rows)
        summary_path = Path(out_dir) / "summary_gmm.csv"
        try:
            summary.to_csv(summary_path, index=False)
            print("Summary saved to", summary_path)
        except Exception as e:
            print("Failed to save summary:", e)

        return summary
    
    def _load_models_list(self, model_paths):
        """
        Принимает None или список путей/объектов и возвращает (models, model_sources).
        Если model_paths is None -> возвращает [self], [None]
        """
        from pathlib import Path
        import joblib

        if model_paths is None:
            return [self], [None]

        models = []
        sources = []
        for p in model_paths:
            # если уже объект модели
            if hasattr(p, "predict_df") and hasattr(p, "gmm"):
                models.append(p)
                sources.append(None)
                continue
            # иначе считаем, что это путь
            try:
                obj = joblib.load(str(p))
                models.append(obj)
                sources.append(str(p))
            except Exception as e:
                print(f"Warning: can't load model {p}: {e}")
        if not models:
            return [self], [None]
        return models, sources


    def plot_gmm_full_figure(self, df, gmm=None, col_x='FITC-A', col_y='FSC-H',
                             title="GMM Full Plot", biexp=False, out_path=None):
        """
        Интегрированная версия plot_gmm_full_figure как метод класса.
        Гарантирует, что цвета компонент одинаковы в обоих сабплотах и подписи
        'infected cells'/'not infected cells' соответствуют right_comp.
        Если gmm is None — используется self.gmm.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path

        if gmm is None:
            if not hasattr(self, "gmm") or self.gmm is None:
                raise RuntimeError("No gmm available for plotting.")
            gmm = self.gmm

        x_raw = df[col_x].to_numpy()
        y_raw = df[col_y].to_numpy()
        labels_raw = df.get("gmm_right", np.full(len(df), -1)).to_numpy()

        mask_valid = np.isfinite(x_raw) & np.isfinite(y_raw) & (x_raw > 0)
        x = x_raw[mask_valid]
        y = y_raw[mask_valid]
        labels = labels_raw[mask_valid].astype(int)

        if x.size == 0:
            raise ValueError("Нет валидных значений для построения.")

        # подготовка для модели (log1p)
        x_pred = np.log1p(x).reshape(-1, 1)

        # сетка в x-space (линейная)
        xs = np.linspace(x.min(), x.max(), 800)
        zs = np.log1p(xs).reshape(-1, 1)   # z = log1p(x)

        # параметры gmm
        n_comp = gmm.n_components
        weights = gmm.weights_.ravel()
        means = gmm.means_.ravel()

        # извлечь дисперсии 1D
        covs = gmm.covariances_
        if covs.ndim == 1:
            vars_ = covs
        elif covs.ndim == 2 and covs.shape[1] == 1:
            vars_ = covs[:, 0]
        else:
            vars_ = covs.reshape(n_comp, -1)[:, 0]
        sigmas = np.sqrt(vars_)

        # pdf в z-space для каждого компонента
        pdfs_z = np.zeros((zs.shape[0], n_comp))
        for k in range(n_comp):
            pdfs_z[:, k] = weights[k] * norm.pdf(zs.ravel(), loc=means[k], scale=sigmas[k])

        # перевод в x-space
        jac = 1.0 / (1.0 + xs)
        pdfs_x = pdfs_z * jac[:, None]
        pdf_mix_x = pdfs_x.sum(axis=1)

        # цвета — одинаковые для всех мест
        colors = plt.cm.tab10(np.arange(n_comp))

        # подписи: если есть self.right_comp, считаем её infected
        if hasattr(self, "right_comp") and self.right_comp is not None and n_comp >= 2:
            right_idx = int(self.right_comp)
            label_names = {}
            for k in range(n_comp):
                if k == right_idx:
                    label_names[k] = "infected cells"
                else:
                    # для 2-компонентов — пометим остальные как not infected
                    label_names[k] = "not infected cells" if n_comp == 2 else f"Comp {k}"
        else:
            # fallback
            label_names = {k: f"Comp {k}" for k in range(n_comp)}

        # рисуем
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Левый: гистограмма + mixture + компоненты
        ax = axes[0]
        ax.hist(x, bins=200, density=True, alpha=0.35, color="gray", label="Data (hist)")
        for k in range(n_comp):
            ax.plot(xs, pdfs_x[:, k], linestyle='--', linewidth=1.8, color=colors[k],
                    label=f"{label_names.get(k, 'Comp '+str(k))} (w={weights[k]:.2f})")
        ax.plot(xs, pdf_mix_x, linewidth=2.2, color="black", label="Mixture")
        ax.set_xlabel(col_x)
        ax.set_xscale("log")
        ax.set_ylabel("Density")
        ax.set_title("Histogram + GMM (x-space)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Правый: scatter по компонентам — используем те же цвета
        ax2 = axes[1]
        if biexp:
            def biexp_transform(v, w=0.5, a=4.5):
                return np.sign(v) * (np.log(1 + w * np.abs(v)) / a)
            x_plot = biexp_transform(x)
            x_label = f"{col_x} (biexp)"
        else:
            x_plot = np.log10(x)
            x_label = f"{col_x} (log10)"

        # метки компонент для точек (используем предсказание gmm на x_pred)
        comp_labels = gmm.predict(np.log1p(x).reshape(-1, 1))
        # цвета точек по component labels, те же colors
        point_colors = colors[comp_labels]

        sc = ax2.scatter(x_plot, y, c=point_colors, s=6, alpha=0.6)
        ax2.set_xlabel(x_label)
        ax2.set_ylabel(col_y)
        ax2.set_title("Scatter colored by GMM component")
        ax2.grid(alpha=0.2)

        # легенда для правого графика: создаём ручки вручную, чтобы подписи совпадали
        from matplotlib.patches import Patch
        legend_patches = []
        used_indices = np.unique(comp_labels)
        for k in used_indices:
            name = label_names.get(int(k), f"Comp {k}")
            legend_patches.append(Patch(color=colors[int(k)], label=f"{name} (comp {int(k)})"))
        ax2.legend(handles=legend_patches, fontsize=9)

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()

        if out_path is not None:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

        # возвращаем stats
        comp_labels_all = gmm.predict(np.log1p(x).reshape(-1, 1))
        total = len(comp_labels_all)
        counts = np.bincount(comp_labels_all, minlength=n_comp)
        stats = [{"comp": i, "count": int(counts[i]), "pct": float(counts[i]/total*100.0)} for i in range(n_comp)]
        return stats


    def process_files_list(self, file_paths, out_dir="gmm_results_list", x_col="FITC-A", y_col="FSC-H",
                           sample_frac_for_plot=0.05, sample_cap=200_000, biexp=False,
                           min_rows=0, model_paths=None, threshold_pct=35.0, summary_plot_name="summary_plot.png"):
        """
        Обработка списка файлов по тому же принципу выбора модели:
          - model_paths: список путей/объектов (если None, используется self)
          - для каждого файла начинаем с первой модели, если infectivity_percent >= threshold_pct
            переключаемся на следующую; принимаем первую модель с infectivity_percent < threshold_pct.
          - сохраняем labeled parquet/labels (в out_dir), строим summary csv и один общий рисунок.
        Возвращает pd.DataFrame summary.
        """
        from pathlib import Path
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # нормализуем file_paths в список Path
        fpaths = [Path(p) for p in file_paths]

        # загрузим модели
        models, model_sources = self._load_models_list(model_paths)

        rows = []
        for fp in fpaths:
            print("Processing:", fp)
            try:
                if fp.suffix.lower() in [".parquet", ".pq"]:
                    df = pd.read_parquet(fp)
                elif fp.suffix.lower() in [".csv", ".txt"]:
                    df = pd.read_csv(fp)
                else:
                    print("  Unsupported file:", fp)
                    rows.append({"file": str(fp), "rows": None, "selected": None, "infectivity_percent": None,
                                 "model_used": "unsupported", "model_path": None})
                    continue
            except Exception as e:
                print("  Read failed:", fp, e)
                rows.append({"file": str(fp), "rows": None, "selected": None, "infectivity_percent": None,
                             "model_used": "read_failed", "model_path": None})
                continue

            total = len(df)
            if total < min_rows:
                print(f"  Skipped {fp.name}: rows {total} < {min_rows}")
                rows.append({"file": str(fp), "rows": total, "selected": 0, "infectivity_percent": None,
                             "model_used": "skipped_min_rows", "model_path": None})
                continue

            accepted = False
            df_labeled_to_save = None
            used_model_idx = None
            used_model_src = None
            stats_for_save = None

            for m_idx, model in enumerate(models):
                try:
                    df_labeled = model.predict_df(df, col=x_col, label_col="gmm_right")
                except Exception as e:
                    print(f"  Model {m_idx} predict failed for {fp.name}: {e}")
                    continue

                selected = int((df_labeled["gmm_right"] == 1).sum())
                infectivity_percent = 100 * selected / total if total > 0 else 0.0

                print(f"   Model {m_idx}: infectivity_percent = {infectivity_percent:.2f}%")

                if infectivity_percent < threshold_pct:
                    accepted = True
                    used_model_idx = m_idx
                    used_model_src = model_sources[m_idx] if m_idx < len(model_sources) else None
                    df_labeled_to_save = df_labeled
                    # component stats if possible
                    x_valid = df_labeled[x_col].to_numpy()
                    x_valid = x_valid[np.isfinite(x_valid) & (x_valid > 0)]
                    if x_valid.size > 0 and hasattr(model, "gmm") and model.gmm is not None:
                        comp = model.gmm.predict(np.log1p(x_valid).reshape(-1, 1))
                        uniq, cnt = np.unique(comp, return_counts=True)
                    else:
                        uniq, cnt = np.array([], dtype=int), np.array([], dtype=int)
                    stats = {
                        "file": str(fp),
                        "rows": total,
                        "selected": int(selected),
                        "infectivity_percent": float(infectivity_percent),
                        "model_used": int(m_idx),
                        "model_path": used_model_src
                    }
                    for u, c in zip(uniq, cnt):
                        stats[f"comp_{u}_count"] = int(c)
                        stats[f"comp_{u}_pct"] = float(c / len(comp) * 100.0) if len(comp)>0 else 0.0
                    stats_for_save = stats
                    break
                else:
                    continue

            if not accepted:
                print(f"  No model produced infectivity_percent < {threshold_pct}%, skipping file.")
                rows.append({"file": str(fp), "rows": total, "selected": 0, "infectivity_percent": None,
                             "model_used": "skipped", "model_path": None})
                continue

            # сохраняем labeled parquet
            out_parquet = out_dir / f"{fp.stem}_gmm.parquet"
            try:
                df_labeled_to_save.to_parquet(out_parquet, index=False)
            except Exception as e:
                print("  Failed to save parquet:", e)

            # сохраняем локальный рисунок для файла (опционально)
            out_png = out_dir / f"{fp.stem}_gmm.png"
            try:
                self.plot_gmm_full_figure(df_labeled_to_save, gmm=models[used_model_idx].gmm,
                                          col_x=x_col, col_y=y_col,
                                          title=f"{fp.name} (model {used_model_idx})",
                                          biexp=biexp, out_path=out_png)
            except Exception as e:
                print("  Plot failed for", fp.name, e)

            rows.append(stats_for_save)

        # общий summary
        summary = pd.DataFrame(rows)
        summary.to_csv(out_dir / "summary_from_list.csv", index=False)
        print("Summary saved to", out_dir / "summary_from_list.csv")

        # --- строим итоговый рисунок: bar plot infectivity_percent по файлам ---
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            # для упрощения — берем только те строки, у которых infectivity_percent не None
            plot_df = summary.copy()
            plot_df['display_name'] = plot_df['file'].apply(lambda p: Path(p).stem if p is not None else str(p))
            # цвета: каждый модельный индекс -> цвет
            unique_models = plot_df['model_used'].fillna('skipped').unique()
            # подготовим палитру
            palette = {}
            cmap = plt.cm.tab10
            model_idxs = [m for m in unique_models if isinstance(m, (int, np.integer))]
            for idx, m in enumerate(sorted(model_idxs)):
                palette[m] = cmap(idx % 10)
            palette['skipped'] = (0.7, 0.7, 0.7)  # grey for skipped/none

            xs = np.arange(len(plot_df))
            heights = plot_df['infectivity_percent'].fillna(0.0).to_numpy()
            colors_bar = [palette.get(v, palette['skipped']) for v in plot_df['model_used'].fillna('skipped')]

            ax.bar(xs, heights, color=colors_bar)
            ax.set_xticks(xs)
            ax.set_xticklabels(plot_df['display_name'], rotation=45, ha='right', fontsize=8)
            ax.set_ylabel("Infectivity percent")
            ax.set_title("Infectivity percent per file (model used indicated by color)")

            # легенда ручками
            from matplotlib.patches import Patch
            legend_handles = []
            for m, col in palette.items():
                name = f"model {m}" if m != 'skipped' else "skipped"
                legend_handles.append(Patch(color=col, label=name))
            ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)

            plt.tight_layout()
            summary_plot_path = out_dir / summary_plot_name
            fig.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("Summary plot saved to", summary_plot_path)
        except Exception as e:
            print("Failed to create summary plot:", e)

        return summary




def plot_gmm_1d_prediction_with_stats(gmm, x, title="GMM 1D Prediction"):
    import numpy as np
    import matplotlib.pyplot as plt

    # --- 1. Очистка данных от NaN и отрицательных ---
    x = np.asarray(x).flatten()
    mask_valid = np.isfinite(x) & (x > 0)
    x_clean = x[mask_valid]

    if len(x_clean) == 0:
        raise ValueError("Нет корректных значений для построения графика (все NaN или <=0).")

    # лог-трансформация как при обучении
    x_log = np.log1p(x_clean).reshape(-1, 1)

    # предсказания
    labels = gmm.predict(x_log)

    # --- 2. Сетка ---
    xs = np.linspace(x_clean.min(), x_clean.max(), 1000)
    xs_log = np.log1p(xs).reshape(-1, 1)

    logprob = gmm.score_samples(xs_log)
    responsibilities = gmm.predict_proba(xs_log)
    pdf = np.exp(logprob)
    pdf0 = responsibilities[:, 0] * pdf
    pdf1 = responsibilities[:, 1] * pdf

    # --- 3. График ---
    plt.figure(figsize=(10, 6))

    # гистограмма
    plt.hist(x_clean, bins=200, density=True, alpha=0.4, color='gray', label="Data")

    # компоненты GMM
    plt.plot(xs, pdf0, linewidth=2, label="Component 0")
    plt.plot(xs, pdf1, linewidth=2, label="Component 1")
    plt.plot(xs, pdf, linewidth=2, label="Mixture", color="black")

    plt.xlabel("FITC-A")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # --- 4. Статистика ---
    print("=== COMPONENT STATISTICS ===")
    for i in range(gmm.means_.shape[0]):
        print(f"Component {i}: mean(log)={gmm.means_[i][0]:.3f}, weight={gmm.weights_[i]:.3f}")

    print("\n=== PREDICTION COUNTS ===")
    uniq, cnt = np.unique(labels, return_counts=True)
    for u, c in zip(uniq, cnt):
        print(f"Label {u}: {c}")

    plt.show()

def plot_gmm_full_figure(df, gmm, col_x='FITC-A', col_y='FSC-H',
                         title="GMM Full Plot", biexp=False, out_path=None):
    """
    Корректно рисует:
      - гистограмму FITC-A (линейная ось)
      - PDF mixture и PDF каждой компоненты, корректно преобразованные из log1p-space в x-space
      - scatter FSC-H vs FITC-A (лог по X или biexp)
    Если out_path задан — сохраняет фигуру туда.
    """
    x_raw = df[col_x].to_numpy()
    y_raw = df[col_y].to_numpy()
    labels_raw = df.get("gmm_right", np.full(len(df), -1)).to_numpy()

    mask_valid = np.isfinite(x_raw) & np.isfinite(y_raw) & (x_raw > 0)
    x = x_raw[mask_valid]
    y = y_raw[mask_valid]
    labels = labels_raw[mask_valid].astype(int)

    if x.size == 0:
        raise ValueError("Нет валидных значений для построения.")

    # x_pred для модели: лог так же как при обучении (log1p)
    x_pred = np.log1p(x).reshape(-1, 1)

    # сетка в x-space (линейная)
    xs = np.linspace(x.min(), x.max(), 800)
    zs = np.log1p(xs).reshape(-1, 1)   # z = log1p(x)

    # --- Получаем pdf в z-space для каждого компонента ---
    # извлекаем параметры (варианс учитываем общий формат covariances_)
    n_comp = gmm.n_components
    weights = gmm.weights_.ravel()
    means = gmm.means_.ravel()

    # извлечь дисперсии 1D
    covs = gmm.covariances_
    if covs.ndim == 1:
        vars_ = covs
    elif covs.ndim == 2 and covs.shape[1] == 1:
        vars_ = covs[:, 0]
    else:
        # возможно форма (n_comp, 1, 1)
        vars_ = covs.reshape(n_comp, -1)[:, 0]

    sigmas = np.sqrt(vars_)

    # pdf_z_k(z) = weight_k * N(z | mu_k, sigma_k)
    pdfs_z = np.zeros((zs.shape[0], n_comp))
    for k in range(n_comp):
        pdfs_z[:, k] = weights[k] * norm.pdf(zs.ravel(), loc=means[k], scale=sigmas[k])

    # переводим в x-space: pdf_x(x) = pdf_z(z) * dz/dx = pdf_z(z) * 1/(1+x)
    jac = 1.0 / (1.0 + xs)  # dz/dx evaluated on xs
    pdfs_x = pdfs_z * jac[:, None]   # shape (len(xs), n_comp)
    pdf_mix_x = pdfs_x.sum(axis=1)

    # --- Рисуем сабплоты ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # левый: гистограмма + mixture + components (в x-space)
    ax = axes[0]
    ax.hist(x, bins=200, density=True, alpha=0.35, color="gray", label="Data (hist)")

    # компонентные кривые
    colors = plt.cm.tab10(np.arange(n_comp))
    for k in range(n_comp):
        ax.plot(xs, pdfs_x[:, k], linestyle='--', linewidth=1.8, color=colors[k],
                label=f"Comp {k} (w={weights[k]:.2f})")

    # смесь
    ax.plot(xs, pdf_mix_x, linewidth=2.2, color="black", label="Mixture")

    ax.set_xlabel(col_x)
    ax.set_xscale("log")
    ax.set_ylabel("Density")
    ax.set_title("Histogram + GMM (x-space)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # правый: scatter FSC-H vs FITC-A with color by component (use same comp labels from model)
    ax2 = axes[1]
    # for plotting we can use log10 or biexp transform for readability
    if biexp:
        def biexp_transform(v, w=0.5, a=4.5):
            return np.sign(v) * (np.log(1 + w * np.abs(v)) / a)
        x_plot = biexp_transform(x)
    else:
        x_plot = np.log10(x)

    sc = ax2.scatter(x_plot, y, c=gmm.predict(x_pred), cmap="coolwarm", s=6, alpha=0.5)
    ax2.set_xlabel(f"{col_x} ({'biexp' if biexp else 'log10'})")
    ax2.set_ylabel(col_y)
    ax2.set_title("Scatter colored by GMM component")
    ax2.grid(alpha=0.2)
    plt.colorbar(sc, ax=ax2, label="component")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    # возвращаем stats
    comp_labels = gmm.predict(np.log1p(x).reshape(-1, 1))
    total = len(comp_labels)
    counts = np.bincount(comp_labels, minlength=n_comp)
    stats = [{"comp": i, "count": int(counts[i]), "pct": float(counts[i]/total*100.0)} for i in range(n_comp)]
    return stats