"""
Klima – TabPFN Hyperparameter Sensitivity Experiment
====================================================
Sweeps the most important TabPFNRegressor knobs on the chronological
NASA POWER clear-sky solar dataset and reports R² / MAE / RMSE.

Usage
-----
    cd klima_v2
    python src/tabpfn_sensitivity.py

Optional environment variables
------------------------------
    TABPFN_TOKEN   – required for real TabPFN weights (see README)
    DEVICE         – "cpu" | "cuda" | "mps"  (default: cpu)
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import load_data, make_features, HAS_TABPFN

if HAS_TABPFN:
    from tabpfn import TabPFNRegressor
else:
    TabPFNRegressor = None
    print("WARNING: tabpfn package not importable – experiment will be skipped.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def metrics(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def prepare_xy(train_ratio: float = 0.85):
    """Load data, engineer features, return chronological X/y splits."""
    data_dir = Path(__file__).resolve().parents[1] / "data"
    df = load_data(data_dir)
    df = make_features(df)

    exclude = {
        "YEAR", "DOY", "MO", "DY", "date", "target", "solar",
        "par", "allsky", "TS", "CLRSKY_SFC_SW_DWN", "CLRSKY_SFC_PAR_TOT",
        "ALLSKY_SFC_SW_DWN",
    }
    feature_cols = [c for c in df.columns if c not in exclude and not str(c).startswith("Unnamed")]

    split = int(len(df) * train_ratio)
    train, test = df.iloc[:split], df.iloc[split:]

    X_train = train[feature_cols].astype(np.float64)
    y_train = train["target"].astype(np.float64)
    X_test = test[feature_cols].astype(np.float64)
    y_test = test["target"].astype(np.float64)

    return X_train, y_train, X_test, y_test, feature_cols


def run_one(name: str, params: dict, X_train, y_train, X_test, y_test, device: str):
    """Fit one TabPFN configuration and return metrics + wall time."""
    if not HAS_TABPFN:
        return {"config": name, "error": "tabpfn not available"}

    # Cap training size for speed / memory (TabPFN is strongest on ≤ a few k rows)
    max_train = 2500
    if len(X_train) > max_train:
        idx = np.linspace(0, len(X_train) - 1, max_train, dtype=int)
        X_fit, y_fit = X_train.iloc[idx], y_train.iloc[idx]
    else:
        X_fit, y_fit = X_train, y_train

    t0 = time.perf_counter()
    try:
        model = TabPFNRegressor(device=device, **params)
        model.fit(X_fit, y_fit)
        pred = model.predict(X_test)
        m = metrics(y_test, pred)
        m["wall_s"] = round(time.perf_counter() - t0, 2)
        m["config"] = name
        m["n_train"] = len(X_fit)
        m["error"] = None
        return m
    except Exception as e:
        return {
            "config": name,
            "error": str(e)[:200],
            "wall_s": round(time.perf_counter() - t0, 2),
            "r2": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
        }


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------
def build_grid(device: str) -> list[tuple[str, dict]]:
    """
    Return (name, kwargs) pairs covering the most important TabPFN knobs.
    Keep the grid modest so a full run finishes in a reasonable time on CPU.
    """
    base = {"random_state": 42}

    grid = [
        # ---- n_estimators sweep (primary lever) ----
        ("n_est=4",  {**base, "n_estimators": 4}),
        ("n_est=8",  {**base, "n_estimators": 8}),          # common default
        ("n_est=16", {**base, "n_estimators": 16}),

        # ---- softmax temperature ----
        ("temp=0.7", {**base, "n_estimators": 8, "softmax_temperature": 0.7}),
        ("temp=0.9", {**base, "n_estimators": 8, "softmax_temperature": 0.9}),  # typical auto
        ("temp=1.0", {**base, "n_estimators": 8, "softmax_temperature": 1.0}),
        ("temp=1.2", {**base, "n_estimators": 8, "softmax_temperature": 1.2}),

        # ---- fit_mode ----
        ("mode=fit_preprocessors", {**base, "n_estimators": 8, "fit_mode": "fit_preprocessors"}),
        ("mode=low_memory",        {**base, "n_estimators": 8, "fit_mode": "low_memory"}),
        # fit_with_cache is more useful when you predict many times; still included
        ("mode=fit_with_cache",    {**base, "n_estimators": 8, "fit_mode": "fit_with_cache"}),

        # ---- inference precision ----
        ("prec=auto",    {**base, "n_estimators": 8, "inference_precision": "auto"}),
        ("prec=float32", {**base, "n_estimators": 8, "inference_precision": "float32"}),
    ]

    # Optional temperature calibration (only if the installed version supports it)
    try:
        grid.append(
            ("calibrate_temp", {
                **base,
                "n_estimators": 8,
                "tuning_config": {"calibrate_temperature": True},
            })
        )
    except Exception:
        pass

    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = os.environ.get("DEVICE", "cpu")
    print(f"Device: {device}")
    print(f"HAS_TABPFN: {HAS_TABPFN}")
    if not HAS_TABPFN:
        print("Install tabpfn and set TABPFN_TOKEN to run the real experiment.")
        return

    print("\nPreparing chronological train/test split …")
    X_train, y_train, X_test, y_test, feature_cols = prepare_xy(train_ratio=0.85)
    print(f"  Features : {len(feature_cols)}")
    print(f"  Train    : {len(X_train)} rows")
    print(f"  Test     : {len(X_test)} rows")
    print(f"  Target   : next-day clear-sky SW (MJ/m²/day)\n")

    grid = build_grid(device)
    results = []

    print(f"{'config':<28} {'R²':>8} {'MAE':>8} {'RMSE':>8} {'time(s)':>8}  note")
    print("-" * 80)

    for name, params in grid:
        # Remove keys that the installed version may not accept
        clean = dict(params)
        res = run_one(name, clean, X_train, y_train, X_test, y_test, device)
        results.append(res)

        if res.get("error"):
            print(f"{name:<28} {'—':>8} {'—':>8} {'—':>8} {res['wall_s']:>8.1f}  ERROR: {res['error'][:50]}")
        else:
            print(
                f"{name:<28} {res['r2']:>8.4f} {res['mae']:>8.3f} "
                f"{res['rmse']:>8.3f} {res['wall_s']:>8.1f}"
            )

    # Summary table
    df = pd.DataFrame(results)
    out_dir = Path(__file__).resolve().parents[1] / "models"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "tabpfn_sensitivity.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults written to {csv_path}")

    # Best config by R²
    valid = df.dropna(subset=["r2"])
    if not valid.empty:
        best = valid.loc[valid["r2"].idxmax()]
        print("\n=== Best configuration (highest R²) ===")
        print(f"  config : {best['config']}")
        print(f"  R²     : {best['r2']:.4f}")
        print(f"  MAE    : {best['mae']:.3f} MJ/m²")
        print(f"  RMSE   : {best['rmse']:.3f} MJ/m²")
        print(f"  time   : {best['wall_s']:.1f} s")

        # Quick ranking
        print("\n=== Ranking by R² ===")
        ranking = valid.sort_values("r2", ascending=False)[["config", "r2", "mae", "wall_s"]]
        print(ranking.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
