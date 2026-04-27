from __future__ import annotations

import warnings
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline as SkPipeline

from features import build_features, get_feature_columns, _clear_cache, SEED

warnings.filterwarnings("ignore")
np.random.seed(SEED)

# Optional dependencies
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("[WARN] pip install lightgbm")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] pip install xgboost")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("[WARN] pip install optuna  (LightGBM sẽ dùng default params)")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[WARN] pip install shap")

# Paths
DATA_DIR = Path("/kaggle/input/competitions/datathon-2026-round-1")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# SECTION 0 — LOG-TRANSFORM UTILITIES
# ============================================================================

def log_transform(series: pd.Series) -> pd.Series:
    return np.log1p(series.clip(lower=0))


def inv_log_transform(series: np.ndarray) -> np.ndarray:
    return np.expm1(np.clip(series, 0, None))

# =============================================================================
# SECTION 1 — DATA LOADING
# =============================================================================

def load_train_data(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "sales.csv", parse_dates=["Date"])
    df = df[df["Date"] < "2023-01-01"].sort_values("Date").reset_index(drop=True)
    print(f"  Train loaded: {len(df)} rows | {df.Date.min().date()} → {df.Date.max().date()}")
    return df


# ============================================================================
# SECTION 2 — TREND FEATURE COLUMNS
# =============================================================================


TREND_FEATURES = [
    # Time structure
    "trend_days",
    "trend_days_sq",
    "sin_year_1", "cos_year_1",
    "sin_year_2", "cos_year_2",
    "sin_year_3", "cos_year_3",
    "sin_week_1", "cos_week_1",
    "sin_week_2", "cos_week_2",
    # Calendar flags
    "is_holiday",
    "is_pre_tet_rush",
    "is_tet_holiday",
    "days_to_tet",
    "is_payday_window",
    "is_weekend",
    # Promotion signals
    "promo_n_active",
    "promo_max_eff_rate",
    # slow-moving demand quality signals
    "rev_avg_rating_7d",
    "geo_hhi_7d",
    "ship_avg_delivery_days_7d",
]


def get_available_trend_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in TREND_FEATURES if c in df.columns]


# ============================================================================
# SECTION 3 — VALIDATION SPLIT
# ============================================================================

VAL_START = "2021-07-01"
VAL_END   = "2022-12-31"


def make_train_val_split(df_model: pd.DataFrame):
    val_mask = (df_model["Date"] >= VAL_START) & (df_model["Date"] <= VAL_END)
    return df_model[~val_mask].copy(), df_model[val_mask].copy()


# ============================================================================
# SECTION 4 — TIMESERIES CV
# ============================================================================

def run_cross_validation(
    X: pd.DataFrame,
    y_log: pd.Series,
    model_fn,
    n_splits: int = 3,
    target_name: str = "Revenue",
) -> tuple[float, list]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rmses, oof_preds = [], []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_vl = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_vl = y_log.iloc[tr_idx], y_log.iloc[val_idx]

        model = model_fn()
        model.fit(X_tr, y_tr)
        preds_log = np.clip(model.predict(X_vl), 0, None)

        rmse = mean_squared_error(y_vl, preds_log) ** 0.5
        fold_rmses.append(rmse)
        oof_preds.append((val_idx, preds_log))
        print(f"    Fold {fold+1}/{n_splits} | log-RMSE: {rmse:.5f}")

    mean_rmse = float(np.mean(fold_rmses))
    print(f"    → {target_name} CV Mean log-RMSE: {mean_rmse:.5f}")
    return mean_rmse, oof_preds


# ============================================================================
# SECTION 5 — HYBRID FORECASTER
# ============================================================================

class HybridForecaster:

    def __init__(self, tree_model_fn, trend_cols: list[str]):
        self.trend_cols    = trend_cols
        self.tree_model_fn = tree_model_fn
        self._ridge = SkPipeline([
            ("sc", StandardScaler()),
            ("rg", Ridge(alpha=500.0, random_state=SEED)),
        ])
        self._tree = None

    def fit(self, X: pd.DataFrame, y_log: pd.Series):
        avail = [c for c in self.trend_cols if c in X.columns]
        self._ridge.fit(X[avail], y_log)
        trend_pred = self._ridge.predict(X[avail])

        residuals  = y_log.values - trend_pred
        self._tree = self.tree_model_fn()
        self._tree.fit(X, residuals)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        avail = [c for c in self.trend_cols if c in X.columns]
        return self._ridge.predict(X[avail]) + np.clip(self._tree.predict(X), -5, 5)

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return inv_log_transform(self.predict(X))


# ============================================================================
# SECTION 6 — MODEL FACTORIES
# ============================================================================

def _lgbm_objective(trial, X_tr, y_tr, X_vl, y_vl):
    params = {
        "objective":         "regression",
        "metric":            "rmse",
        "verbosity":         -1,
        "random_state":      SEED,
        "n_estimators":      trial.suggest_int("n_estimators", 400, 2000),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 127),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
        "feature_fraction":  trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq":      trial.suggest_int("bagging_freq", 1, 10),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }
    m = lgb.LGBMRegressor(**params)
    m.fit(X_tr, y_tr,
          eval_set=[(X_vl, y_vl)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    return (mean_squared_error(y_vl, m.predict(X_vl)) ** 0.5)


def tune_lgbm(X_train, y_log_train, n_trials=50, target_name="Revenue") -> dict:
    if not HAS_OPTUNA:
        return {
            "objective": "regression", "metric": "rmse", "verbosity": -1,
            "n_estimators": 1000, "learning_rate": 0.02, "num_leaves": 63,
            "feature_fraction": 0.7, "bagging_fraction": 0.8,
            "bagging_freq": 5, "reg_alpha": 0.1, "reg_lambda": 0.1,
            "random_state": SEED,
        }

    split = int(len(X_train) * 0.8)
    X_tr, X_vl = X_train.iloc[:split], X_train.iloc[split:]
    y_tr, y_vl = y_log_train.iloc[:split], y_log_train.iloc[split:]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(
        lambda t: _lgbm_objective(t, X_tr, y_tr, X_vl, y_vl),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    best = study.best_params
    best.update({"objective": "regression", "metric": "rmse",
                 "verbosity": -1, "random_state": SEED})
    print(f"    [Optuna-{target_name}] best log-RMSE={study.best_value:.5f}")
    return best


def make_lgbm_factory(params: dict):
    def factory():
        return lgb.LGBMRegressor(**params)
    return factory


def make_xgb_factory():
    def factory():
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1000, learning_rate=0.02, max_depth=6,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            tree_method="hist", random_state=SEED, verbosity=0,
        )
    return factory


# ============================================================================
# SECTION 7 — ENSEMBLE
# ============================================================================

def compute_ensemble_weights(cv_rmses: dict[str, float]) -> dict[str, float]:
    inv   = {k: 1.0 / v for k, v in cv_rmses.items()}
    total = sum(inv.values())
    w     = {k: v / total for k, v in inv.items()}
    print("\n  ── Ensemble Weights ──")
    for name, weight in w.items():
        print(f"    {name:<16s}: {weight:.4f}  (CV log-RMSE = {cv_rmses[name]:.5f})")
    return w


def ensemble_predict_log(models: dict, weights: dict, X: pd.DataFrame) -> np.ndarray:
    blend = np.zeros(len(X), dtype=np.float64)
    for name, model in models.items():
        blend += weights[name] * model.predict(X)
    return blend


def ensemble_predict_raw(models: dict, weights: dict, X: pd.DataFrame) -> np.ndarray:
    return inv_log_transform(ensemble_predict_log(models, weights, X))


# ============================================================================
# SECTION 8 — EVALUATION & PLOTTING
# ============================================================================

def evaluate_predictions(y_true_raw, y_pred_raw, target_name: str) -> dict:
    mae  = mean_absolute_error(y_true_raw, y_pred_raw)
    rmse = mean_squared_error(y_true_raw, y_pred_raw) ** 0.5
    r2   = r2_score(y_true_raw, y_pred_raw)
    print(f"\n  ── {target_name} — Val Metrics (2021-07 → 2022-12) ──")
    print(f"     MAE   : {mae:>15,.0f}")
    print(f"     RMSE  : {rmse:>15,.0f}")
    print(f"     R²    : {r2:>15.4f}")
    return {"target": target_name, "MAE": mae, "RMSE": rmse, "R2": r2}


def plot_validation(dates, y_true, y_pred, target_name, output_dir):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates, y_true, lw=0.8, label="Actual",    color="steelblue")
    ax.plot(dates, y_pred, lw=0.8, label="Predicted", color="tomato", linestyle="--")
    ax.set_title(f"{target_name} — Actual vs Predicted (2021-07→2022-12)", fontsize=13)
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    plt.tight_layout()
    p = output_dir / f"val_plot_{target_name.lower()}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {p}")


# ============================================================================
# SECTION 9 — SHAP ANALYSIS
# ============================================================================

def run_shap_analysis(model, X_sample, target_name, output_dir, top_n=25):
    if not HAS_SHAP:
        print("  [SHAP] Skip — pip install shap")
        return

    inner = model._tree if isinstance(model, HybridForecaster) else model
    print(f"\n  [SHAP] TreeExplainer → {target_name} (log residuals) …")

    explainer   = shap.TreeExplainer(inner)
    shap_values = explainer.shap_values(X_sample)

    fig, _ = plt.subplots(figsize=(10, 9))
    shap.summary_plot(shap_values, X_sample, plot_type="bar",
                      max_display=top_n, show=False, color="steelblue")
    plt.title(f"SHAP Feature Importance — {target_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p = output_dir / f"shap_{target_name.lower()}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {p}")


# ============================================================================
# SECTION 10 — WALK-FORWARD FORECAST
# ============================================================================

def iterative_forecast(
    models_rev:  dict,
    models_cogs: dict,
    weights_rev: dict,
    weights_cogs: dict,
    df_history:  pd.DataFrame,
    test_dates:  pd.DatetimeIndex,
    feat_cols:   list[str],
    data_dir:    Path,
) -> pd.DataFrame:

    print("\n  [Forecast] Walk-forward inference …")
    history = df_history[["Date", "Revenue", "COGS"]].copy()
    results = []

    for i, dt in enumerate(test_dates):
        if (i + 1) % 50 == 0:
            print(f"    Day {i+1}/{len(test_dates)}: {dt.date()}")

        placeholder = pd.DataFrame([{"Date": dt}])
        combined    = pd.concat([history, placeholder], ignore_index=True)

        # build_features reads auxiliary CSVs from cache — no disk I/O per step
        feat_df  = build_features(combined, data_dir=data_dir, is_test=True)
        last_row = feat_df.iloc[[-1]]

        avail  = [c for c in feat_cols if c in last_row.columns]
        X_row  = last_row[avail].fillna(0)

        rev_pred  = float(ensemble_predict_raw(models_rev,  weights_rev,  X_row)[0])
        cogs_pred = float(ensemble_predict_raw(models_cogs, weights_cogs, X_row)[0])

        results.append({"Date": dt, "Revenue": rev_pred, "COGS": cogs_pred})

        history = pd.concat(
            [history,
             pd.DataFrame([{"Date": dt, "Revenue": rev_pred, "COGS": cogs_pred}])],
            ignore_index=True,
        )

    return pd.DataFrame(results)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "═" * 70)
    print("Team ThirdofJune")
    print("═" * 70 + "\n")

    if not HAS_LGBM:
        raise RuntimeError("pip install lightgbm")
    if not HAS_XGB:
        raise RuntimeError("pip install xgboost")

    # Flush any stale cached CSVs from previous runs
    _clear_cache()

    # STEP 1: Feature Engineering
    print("STEP 1 — Feature Engineering (includes reviews / shipments / returns / geography)\n" + "─" * 60)
    df_raw  = load_train_data(DATA_DIR)
    df_feat = build_features(df_raw, data_dir=DATA_DIR, is_test=False)

    df_model = df_feat.dropna(subset=["Revenue_lag365", "COGS_lag365"]).copy()
    feat_cols  = get_feature_columns(df_model)
    trend_cols = get_available_trend_cols(df_model)

    print(f"  Usable rows : {len(df_model)}")
    print(f"  Total feats : {len(feat_cols)}")
    print(f"  Trend feats : {len(trend_cols)}")

    new_feat_groups = {
        "Review":    [c for c in feat_cols if c.startswith("rev_")],
        "Shipment":  [c for c in feat_cols if c.startswith("ship_")],
        "Return":    [c for c in feat_cols if c.startswith("ret_")],
        "Geography": [c for c in feat_cols if c.startswith("geo_")],
    }
    for group, cols in new_feat_groups.items():
        print(f"    {group:10s}: {len(cols)} features — {cols[:4]} …")

    # STEP 2: Log-transform targets
    print("\nSTEP 2 — Log1p Transform\n" + "─" * 40)
    df_model["log_Revenue"] = log_transform(df_model["Revenue"])
    df_model["log_COGS"]    = log_transform(df_model["COGS"])
    print(f"  log_Revenue | mean={df_model['log_Revenue'].mean():.3f}  "
          f"std={df_model['log_Revenue'].std():.3f}")

    # STEP 3: Train / Val split
    print("\nSTEP 3 — Train/Val Split (mirrors 18-month test window)\n" + "─" * 40)
    train_df, val_df = make_train_val_split(df_model)
    X_tr_full  = train_df[feat_cols].fillna(0)
    X_val_full = val_df[feat_cols].fillna(0)
    print(f"  Train: {len(train_df)} rows  ({train_df.Date.min().date()} → {train_df.Date.max().date()})")
    print(f"  Val  : {len(val_df)} rows  ({val_df.Date.min().date()} → {val_df.Date.max().date()})")

    # STEP 4: Optuna Tuning
    print("\nSTEP 4 — Optuna Tuning (log-space)\n" + "─" * 40)
    params_rev  = tune_lgbm(X_tr_full, train_df["log_Revenue"], n_trials=50, target_name="Revenue")
    params_cogs = tune_lgbm(X_tr_full, train_df["log_COGS"],    n_trials=50, target_name="COGS")
    with open(OUTPUT_DIR / "lgbm_best_params.json", "w") as f:
        json.dump({"Revenue": params_rev, "COGS": params_cogs}, f, indent=2)

    # STEP 5: TimeSeriesSplit CV
    print("\nSTEP 5 — TimeSeriesSplit CV (n_splits=3)\n" + "─" * 40)
    X_cv       = df_model[feat_cols].fillna(0)
    y_rev_log  = df_model["log_Revenue"]
    y_cogs_log = df_model["log_COGS"]

    cv_rev, cv_cogs = {}, {}

    def make_hybrid_lgbm_rev():
        return HybridForecaster(make_lgbm_factory(params_rev), trend_cols)
    def make_hybrid_lgbm_cogs():
        return HybridForecaster(make_lgbm_factory(params_cogs), trend_cols)
    def make_hybrid_xgb():
        return HybridForecaster(make_xgb_factory(), trend_cols)

    print("\n  [Hybrid-LightGBM] Revenue:")
    cv_rev["hybrid_lgbm"],  _ = run_cross_validation(X_cv, y_rev_log, make_hybrid_lgbm_rev, 3, "Revenue")
    print("\n  [Hybrid-LightGBM] COGS:")
    cv_cogs["hybrid_lgbm"], _ = run_cross_validation(X_cv, y_cogs_log, make_hybrid_lgbm_cogs, 3, "COGS")
    print("\n  [Hybrid-XGBoost] Revenue:")
    cv_rev["hybrid_xgb"],   _ = run_cross_validation(X_cv, y_rev_log, make_hybrid_xgb, 3, "Revenue")
    print("\n  [Hybrid-XGBoost] COGS:")
    cv_cogs["hybrid_xgb"],  _ = run_cross_validation(X_cv, y_cogs_log, make_hybrid_xgb, 3, "COGS")

    # STEP 6: Train Final Models
    print("\nSTEP 6 — Train Final Models\n" + "─" * 40)

    hybrid_lgbm_rev  = HybridForecaster(make_lgbm_factory(params_rev),  trend_cols)
    hybrid_lgbm_cogs = HybridForecaster(make_lgbm_factory(params_cogs), trend_cols)
    hybrid_xgb_rev   = HybridForecaster(make_xgb_factory(), trend_cols)
    hybrid_xgb_cogs  = HybridForecaster(make_xgb_factory(), trend_cols)

    hybrid_lgbm_rev.fit(X_tr_full,  train_df["log_Revenue"])
    hybrid_lgbm_cogs.fit(X_tr_full, train_df["log_COGS"])
    hybrid_xgb_rev.fit(X_tr_full,   train_df["log_Revenue"])
    hybrid_xgb_cogs.fit(X_tr_full,  train_df["log_COGS"])

    models_rev  = {"hybrid_lgbm": hybrid_lgbm_rev,  "hybrid_xgb": hybrid_xgb_rev}
    models_cogs = {"hybrid_lgbm": hybrid_lgbm_cogs, "hybrid_xgb": hybrid_xgb_cogs}

    print("\n  Revenue weights:")
    w_rev  = compute_ensemble_weights(cv_rev)
    print("\n  COGS weights:")
    w_cogs = compute_ensemble_weights(cv_cogs)

    # STEP 7: Validate
    print("\nSTEP 7 — Evaluate (2021-07 → 2022-12)\n" + "─" * 40)
    pred_rev_raw  = ensemble_predict_raw(models_rev,  w_rev,  X_val_full)
    pred_cogs_raw = ensemble_predict_raw(models_cogs, w_cogs, X_val_full)

    m_rev  = evaluate_predictions(val_df["Revenue"], pred_rev_raw,  "Revenue")
    m_cogs = evaluate_predictions(val_df["COGS"],    pred_cogs_raw, "COGS")
    plot_validation(val_df["Date"], val_df["Revenue"], pred_rev_raw,  "Revenue", OUTPUT_DIR)
    plot_validation(val_df["Date"], val_df["COGS"],    pred_cogs_raw, "COGS",    OUTPUT_DIR)

    # STEP 8: SHAP
    print("\nSTEP 8 — SHAP Analysis\n" + "─" * 40)
    shap_sample = X_val_full.sample(min(300, len(X_val_full)), random_state=SEED)
    run_shap_analysis(hybrid_lgbm_rev,  shap_sample, "Revenue", OUTPUT_DIR)
    run_shap_analysis(hybrid_lgbm_cogs, shap_sample, "COGS",    OUTPUT_DIR)

    # STEP 9: Walk-Forward Forecast
    print("\nSTEP 9 — Walk-Forward Forecast (2023-01-01 → 2024-07-01)\n" + "─" * 40)
    test_dates = pd.date_range("2023-01-01", "2024-07-01", freq="D")
    submission = iterative_forecast(
        models_rev, models_cogs, w_rev, w_cogs,
        df_raw, test_dates, feat_cols, DATA_DIR,
    )

    # STEP 10: Export
    print("\nSTEP 10 — Export\n" + "─" * 40)
    submission["Date"]    = pd.to_datetime(submission["Date"]).dt.strftime("%Y-%m-%d")
    submission["Revenue"] = submission["Revenue"].round(2)
    submission["COGS"]    = submission["COGS"].round(2)
    submission            = submission[["Date", "Revenue", "COGS"]]
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)

    pd.DataFrame([m_rev, m_cogs]).to_csv(
        OUTPUT_DIR / "validation_metrics.csv", index=False)

    print(f"\n{'═' * 70}")
    print(f"  submission.csv  → {OUTPUT_DIR / 'submission.csv'}")
    print(f"  Rows: {len(submission)}  |  {submission.Date.iloc[0]} → {submission.Date.iloc[-1]}")
    print(f"{'═' * 70}\n")
    return submission


if __name__ == "__main__":
    main()
