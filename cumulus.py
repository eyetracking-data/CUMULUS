from __future__ import annotations

import argparse
from pathlib import Path
import traceback
import numpy as np
import pandas as pd

from scipy.stats import wilcoxon, ks_2samp, zscore, median_abs_deviation, skew, kurtosis

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FEATURE_COLS = ["Gaze X", "Gaze Y", "ET_PupilLeft", "ET_PupilRight"]

MISSING_LEVELS = [5, 10, 15, 20]
OUTLIER_LEVELS = [5, 10, 15, 20]

COLUMN_ALIASES: dict[str, list[str]] = {
    # Example:
    # "Gaze X": ["GazeX", "Gaze_X", "GazePointX"],
    # "Gaze Y": ["GazeY", "Gaze_Y", "GazePointY"],
    # "ET_PupilLeft": ["PupilLeft", "PupilDiameterLeft"],
    # "ET_PupilRight": ["PupilRight", "PupilDiameterRight"],
}

SCALERS = {
    "minmax": MinMaxScaler(),
    "zscore": StandardScaler(),
    "robust": RobustScaler(),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CUMULUS benchmark for eye-tracking preprocessing pipelines."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="One or more dataset directories containing CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where Excel result files will be written.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively for CSV files in dataset directories.",
    )
    parser.add_argument(
        "--enable-gpr",
        action="store_true",
        help="Enable Gaussian Process Regression imputation (slow).",
    )
    parser.add_argument(
        "--enable-mice",
        action="store_true",
        help="Enable MICE / IterativeImputer (slow).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def find_csv_files(root: Path, recursive: bool = True) -> list[Path]:
    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted([p for p in root.glob(pattern) if p.is_file()])


def resolve_feature_columns(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    cols = list(df.columns)
    resolved = []
    for w in wanted:
        if w in cols:
            resolved.append(w)
            continue

        found = None
        for alt in COLUMN_ALIASES.get(w, []):
            if alt in cols:
                found = alt
                break

        if found is None:
            raise KeyError(
                f"Missing column '{w}'. Available example columns: {cols[:20]}"
            )
        resolved.append(found)

    return resolved


def load_eye_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    use_cols = resolve_feature_columns(df, FEATURE_COLS)
    df = df[use_cols].copy()
    rename = {use_cols[i]: FEATURE_COLS[i] for i in range(len(FEATURE_COLS))}
    df = df.rename(columns=rename)
    return df


def safe_wilcoxon(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    common = a.index.intersection(b.index)
    if len(common) < 5:
        return np.nan, np.nan

    aa = a.loc[common].astype(float)
    bb = b.loc[common].astype(float)

    if np.allclose((aa - bb).to_numpy(), 0, equal_nan=True):
        return 0.0, 1.0

    try:
        stat, p = wilcoxon(aa, bb)
        return stat, p
    except Exception:
        return np.nan, np.nan


def introduce_missing_values_mcar(
    df: pd.DataFrame, cols: list[str], missing_percent: float, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)

    for col in cols:
        k = int(n * missing_percent / 100.0)
        if k <= 0:
            continue
        idx = rng.choice(out.index.to_numpy(), size=k, replace=False)
        out.loc[idx, col] = np.nan

    return out


def introduce_outliers_additive(
    df: pd.DataFrame,
    cols: list[str],
    outlier_percent: float,
    scale_range: tuple[float, float] = (5.0, 10.0),
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)

    for col in cols:
        k = int(n * outlier_percent / 100.0)
        if k <= 0:
            continue

        idx = rng.choice(out.index.to_numpy(), size=k, replace=False)
        std = np.nanstd(out[col].to_numpy())
        if std == 0 or np.isnan(std):
            continue

        mult = rng.uniform(scale_range[0], scale_range[1], size=k)
        out.loc[idx, col] = out.loc[idx, col] + mult * std

    return out


def mean_imputation(df: pd.DataFrame) -> pd.DataFrame:
    imp = SimpleImputer(strategy="mean")
    return pd.DataFrame(imp.fit_transform(df), columns=df.columns, index=df.index)


def locf_imputation(df: pd.DataFrame) -> pd.DataFrame:
    return df.ffill().bfill()


def knn_imputation(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    imp = KNNImputer(n_neighbors=k)
    return pd.DataFrame(imp.fit_transform(df), columns=df.columns, index=df.index)


def gpr_imputation(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    x = np.arange(len(df)).reshape(-1, 1)

    for c in df.columns:
        y = df[c].to_numpy(dtype=float)
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            out.append(pd.Series(y, index=df.index, name=c))
            continue

        kernel = C(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-2, 1e2))
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=2,
            random_state=RANDOM_SEED,
        )
        gp.fit(x[mask], y[mask])

        y2 = y.copy()
        miss = ~mask
        if miss.any():
            y2[miss] = gp.predict(x[miss])

        out.append(pd.Series(y2, index=df.index, name=c))

    return pd.concat(out, axis=1)


def mice_imputation(df: pd.DataFrame) -> pd.DataFrame:
    imp = IterativeImputer(max_iter=10, random_state=RANDOM_SEED)
    return pd.DataFrame(imp.fit_transform(df), columns=df.columns, index=df.index)


def get_imputers(enable_gpr: bool, enable_mice: bool):
    imputers = {
        "mean": mean_imputation,
        "locf": locf_imputation,
        "knn(k=5)": lambda d: knn_imputation(d, k=5),
    }
    if enable_gpr:
        imputers["gpr"] = gpr_imputation
    if enable_mice:
        imputers["mice"] = mice_imputation
    return imputers


def remove_and_reconstruct(
    original: pd.DataFrame, imputer_fn, missing_percent: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    masked = introduce_missing_values_mcar(
        original, list(original.columns), missing_percent, seed=seed
    )
    imputed = imputer_fn(masked)
    return masked, imputed


def eval_imputation(
    original: pd.DataFrame, masked: pd.DataFrame, imputed: pd.DataFrame
) -> pd.DataFrame:
    rows = []

    for col in original.columns:
        mask_positions = masked[col].isna() & original[col].notna()
        y_true = original.loc[mask_positions, col]
        y_hat = imputed.loc[mask_positions, col]

        if len(y_true) == 0:
            continue

        rmse = mean_squared_error(y_true, y_hat, squared=False)
        mae = mean_absolute_error(y_true, y_hat)

        o = original[col].dropna()
        p = imputed.loc[o.index, col].dropna()
        w_stat, w_p = safe_wilcoxon(o, p)

        rows.append(
            {
                "feature": col,
                "rmse": rmse,
                "mae": mae,
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": w_p,
            }
        )

    return pd.DataFrame(rows)


def detect_outliers_to_nan(
    df: pd.DataFrame,
    method: str,
    z_thresh: float = 3.0,
    mad_thresh: float = 3.0,
    contamination: float = 0.05,
) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        x = out[col].to_numpy(dtype=float)

        if method == "zscore":
            fill = np.nanmedian(x)
            zz = np.abs(zscore(pd.Series(x).fillna(fill)))
            out[col] = np.where(zz > z_thresh, np.nan, x)

        elif method == "mad":
            med = np.nanmedian(x)
            mad = median_abs_deviation(pd.Series(x).dropna(), nan_policy="omit")
            if mad == 0 or np.isnan(mad):
                continue
            out[col] = np.where(np.abs(x - med) > mad_thresh * mad, np.nan, x)

        elif method == "iforest":
            mask = ~np.isnan(x)
            if mask.sum() < 10:
                continue

            vals = x[mask].reshape(-1, 1)
            iso = IsolationForest(
                contamination=contamination,
                random_state=RANDOM_SEED,
            )
            pred = iso.fit_predict(vals)

            x2 = x.copy()
            x2[mask] = np.where(pred == -1, np.nan, vals.flatten())
            out[col] = x2

        else:
            raise ValueError(f"Unknown outlier method: {method}")

    return out


def eval_outlier_handling(original: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in original.columns:
        o = original[col].dropna()
        c = cleaned[col].dropna()
        if len(o) < 5 or len(c) < 5:
            continue

        o_var = np.var(o)
        c_var = np.var(c)
        var_red_pct = np.nan if o_var == 0 else (o_var - c_var) / o_var * 100.0
        ks_stat, ks_p = ks_2samp(o, c)

        rows.append(
            {
                "feature": col,
                "variance_reduction_pct": var_red_pct,
                "ks_stat": ks_stat,
                "ks_p": ks_p,
            }
        )

    return pd.DataFrame(rows)


def apply_scaler(df: pd.DataFrame, scaler) -> pd.DataFrame:
    arr = scaler.fit_transform(df.to_numpy(dtype=float))
    return pd.DataFrame(arr, columns=df.columns, index=df.index)


def eval_normalization(pre: pd.DataFrame, post: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in pre.columns:
        a = pre[col].dropna()
        b = post[col].dropna()
        common = a.index.intersection(b.index)
        if len(common) < 5:
            continue

        a = a.loc[common].astype(float)
        b = b.loc[common].astype(float)

        a_var = np.var(a)
        b_var = np.var(b)
        var_red_pct = np.nan if a_var == 0 else (a_var - b_var) / a_var * 100.0

        sk_red = abs(skew(a)) - abs(skew(b))
        ku_red = abs(kurtosis(a)) - abs(kurtosis(b))

        ks_stat, ks_p = ks_2samp(a, b)

        rows.append(
            {
                "feature": col,
                "variance_reduction_pct": var_red_pct,
                "skewness_reduction": sk_red,
                "kurtosis_reduction": ku_red,
                "ks_stat": ks_stat,
                "ks_p": ks_p,
            }
        )

    return pd.DataFrame(rows)


def run_for_file(
    df: pd.DataFrame,
    file_id: str,
    enable_gpr: bool,
    enable_mice: bool,
    seed: int,
) -> dict[str, pd.DataFrame]:
    imp_rows = []
    imputers = get_imputers(enable_gpr=enable_gpr, enable_mice=enable_mice)

    for lvl in MISSING_LEVELS:
        for name, fn in imputers.items():
            masked, imputed = remove_and_reconstruct(df, fn, lvl, seed=seed)
            res = eval_imputation(df, masked, imputed)
            if len(res):
                res.insert(0, "file", file_id)
                res.insert(1, "missing_level", lvl)
                res.insert(2, "method", name)
                imp_rows.append(res)

    imputation_raw = pd.concat(imp_rows, ignore_index=True) if imp_rows else pd.DataFrame()

    out_rows = []
    for lvl in OUTLIER_LEVELS:
        corrupted = introduce_outliers_additive(
            df, list(df.columns), lvl, seed=seed
        )
        for m in ["zscore", "mad", "iforest"]:
            cleaned = detect_outliers_to_nan(corrupted, m)
            res = eval_outlier_handling(df, cleaned)
            if len(res):
                res.insert(0, "file", file_id)
                res.insert(1, "outlier_level", lvl)
                res.insert(2, "method", m)
                out_rows.append(res)

    outlier_raw = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()

    norm_rows = []
    mv_candidates = {
        "knn(k=5)": lambda d: knn_imputation(d, k=5),
        "locf": locf_imputation,
    }

    for mv_name, mv_fn in mv_candidates.items():
        mv_done = mv_fn(df)

        for out_m in ["mad", "iforest"]:
            out_done = detect_outliers_to_nan(
                mv_done,
                "mad" if out_m == "mad" else "iforest",
            )

            pre = locf_imputation(out_done)

            for sc_name, sc in SCALERS.items():
                post = apply_scaler(pre, sc)
                res = eval_normalization(pre, post)
                if len(res):
                    res.insert(0, "file", file_id)
                    res.insert(1, "mv_method", mv_name)
                    res.insert(2, "outlier_method", out_m)
                    res.insert(3, "scaler", sc_name)
                    norm_rows.append(res)

    normalization_raw = (
        pd.concat(norm_rows, ignore_index=True) if norm_rows else pd.DataFrame()
    )

    return {
        "imputation_raw": imputation_raw,
        "outlier_raw": outlier_raw,
        "normalization_raw": normalization_raw,
    }


def summarize_results(
    raw: pd.DataFrame, group_cols: list[str], metric_cols: list[str]
) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    agg = {m: "mean" for m in metric_cols}
    return raw.groupby(group_cols, as_index=False).agg(agg)


def run_dataset(
    dataset_dir: str,
    output_dir: Path,
    recursive: bool,
    enable_gpr: bool,
    enable_mice: bool,
    seed: int,
) -> Path:
    root = Path(dataset_dir)
    files = find_csv_files(root, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {root}")

    log_rows = []
    imp_all, out_all, norm_all = [], [], []

    for p in files:
        file_id = str(p.relative_to(root))
        print(f"[RUN] {root.name} :: {file_id}")

        try:
            df = load_eye_file(p)
            df = df.apply(pd.to_numeric, errors="coerce")

            res = run_for_file(
                df=df,
                file_id=file_id,
                enable_gpr=enable_gpr,
                enable_mice=enable_mice,
                seed=seed,
            )

            if not res["imputation_raw"].empty:
                imp_all.append(res["imputation_raw"])
            if not res["outlier_raw"].empty:
                out_all.append(res["outlier_raw"])
            if not res["normalization_raw"].empty:
                norm_all.append(res["normalization_raw"])

            log_rows.append(
                {
                    "file": file_id,
                    "status": "ok",
                    "rows": len(df),
                    "note": "",
                }
            )

        except Exception as e:
            log_rows.append(
                {
                    "file": file_id,
                    "status": "error",
                    "rows": np.nan,
                    "note": str(e),
                }
            )
            print(f"[ERROR] {root.name} :: {file_id}")
            print(traceback.format_exc())

    imputation_raw = pd.concat(imp_all, ignore_index=True) if imp_all else pd.DataFrame()
    outlier_raw = pd.concat(out_all, ignore_index=True) if out_all else pd.DataFrame()
    normalization_raw = (
        pd.concat(norm_all, ignore_index=True) if norm_all else pd.DataFrame()
    )
    run_log = pd.DataFrame(log_rows)

    imputation_summary = summarize_results(
        imputation_raw,
        group_cols=["missing_level", "method", "feature"],
        metric_cols=["rmse", "mae", "wilcoxon_p"],
    )
    outlier_summary = summarize_results(
        outlier_raw,
        group_cols=["outlier_level", "method", "feature"],
        metric_cols=["variance_reduction_pct", "ks_stat", "ks_p"],
    )
    normalization_summary = summarize_results(
        normalization_raw,
        group_cols=["mv_method", "outlier_method", "scaler", "feature"],
        metric_cols=[
            "variance_reduction_pct",
            "skewness_reduction",
            "kurtosis_reduction",
            "ks_stat",
            "ks_p",
        ],
    )

    out_path = output_dir / f"CUMULUS_{root.name}.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        run_log.to_excel(xw, sheet_name="run_log", index=False)

        imputation_raw.to_excel(xw, sheet_name="imputation_raw", index=False)
        imputation_summary.to_excel(xw, sheet_name="imputation_summary", index=False)

        outlier_raw.to_excel(xw, sheet_name="outlier_raw", index=False)
        outlier_summary.to_excel(xw, sheet_name="outlier_summary", index=False)

        normalization_raw.to_excel(xw, sheet_name="normalization_raw", index=False)
        normalization_summary.to_excel(
            xw, sheet_name="normalization_summary", index=False
        )

    return out_path


def main() -> None:
    args = parse_args()

    global RANDOM_SEED
    RANDOM_SEED = args.random_seed
    np.random.seed(RANDOM_SEED)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created = []

    for dataset in args.datasets:
        print(f"[DATASET] {dataset}")
        xlsx = run_dataset(
            dataset_dir=dataset,
            output_dir=output_dir,
            recursive=args.recursive,
            enable_gpr=args.enable_gpr,
            enable_mice=args.enable_mice,
            seed=args.random_seed,
        )
        print(f"[DONE] wrote: {xlsx}")
        created.append(xlsx)

    print("\nCreated files:")
    for p in created:
        print(f" - {p}")


if __name__ == "__main__":
    main()