"""
Klima – SHAP Feature Importance Analysis
========================================
Computes SHAP values for the Random Forest (TreeExplainer) and Ridge
(LinearExplainer) models on the chronological NASA POWER solar dataset.

Usage
-----
    cd klima_v2
    python src/shap_analysis.py

Outputs
-------
- Console ranking of features by mean |SHAP|
- models/shap_rf_summary.png          (beeswarm)
- models/shap_rf_bar.png              (mean |SHAP| bar)
- models/shap_importance.csv          (numeric table)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import load_data, make_features


# ---------------------------------------------------------------------------
# Data preparation (mirrors pipeline chronological split)
# ---------------------------------------------------------------------------
def prepare(train_ratio: float = 0.85):
    data_dir = Path(__file__).resolve().parents[1] / "data"
    df = load_data(data_dir)
    df = make_features(df)

    exclude = {
        "YEAR", "DOY", "MO", "DY", "date", "target", "solar",
        "par", "allsky", "TS", "CLRSKY_SFC_SW_DWN", "CLRSKY_SFC_PAR_TOT",
        "ALLSKY_SFC_SW_DWN",
    }
    feature_cols = [
        c for c in df.columns
        if c not in exclude and not str(c).startswith("Unnamed")
    ]

    split = int(len(df) * train_ratio)
    train, test = df.iloc[:split], df.iloc[split:]

    X_train = train[feature_cols].astype(np.float64)
    y_train = train["target"].astype(np.float64)
    X_test = test[feature_cols].astype(np.float64)
    y_test = test["target"].astype(np.float64)

    return X_train, y_train, X_test, y_test, feature_cols


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------
def fit_models(X_train, y_train):
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=0.5))
    ridge.fit(X_train, y_train)

    return rf, ridge


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------
def shap_random_forest(rf, X_test, feature_cols, out_dir: Path, max_samples: int = 300):
    """TreeExplainer – exact and fast for Random Forest."""
    # Subsample test set for readable plots / reasonable runtime
    if len(X_test) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_test), max_samples, replace=False)
        X_explain = X_test.iloc[idx]
    else:
        X_explain = X_test

    print(f"  Computing TreeExplainer SHAP on {len(X_explain)} rows …")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_explain)

    # Mean absolute SHAP → global importance
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # --- Beeswarm ---
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_cols,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    beeswarm_path = out_dir / "shap_rf_summary.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {beeswarm_path}")

    # --- Bar plot ---
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_cols,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    bar_path = out_dir / "shap_rf_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {bar_path}")

    return importance, shap_values, X_explain


def shap_ridge(ridge, X_train, X_test, feature_cols, out_dir: Path, max_samples: int = 300):
    """LinearExplainer on the standardized Ridge pipeline."""
    # Extract the fitted scaler + linear model
    scaler = ridge.named_steps["standardscaler"]
    linear = ridge.named_steps["ridge"]

    if len(X_test) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_test), max_samples, replace=False)
        X_explain = X_test.iloc[idx]
    else:
        X_explain = X_test

    X_train_s = scaler.transform(X_train)
    X_explain_s = scaler.transform(X_explain)

    print(f"  Computing LinearExplainer SHAP on {len(X_explain)} rows …")
    explainer = shap.LinearExplainer(linear, X_train_s)
    shap_values = explainer.shap_values(X_explain_s)

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # Coefficient magnitude (for reference)
    coefs = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": linear.coef_,
        "abs_coef": np.abs(linear.coef_),
    }).sort_values("abs_coef", ascending=False)

    # Simple bar of mean |SHAP|
    plt.figure(figsize=(8, 6))
    top = importance.head(20).iloc[::-1]
    plt.barh(top["feature"], top["mean_abs_shap"], color="#2563eb")
    plt.xlabel("mean |SHAP value|")
    plt.title("Ridge – SHAP feature importance")
    plt.tight_layout()
    bar_path = out_dir / "shap_ridge_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {bar_path}")

    return importance, coefs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    out_dir = Path(__file__).resolve().parents[1] / "models"
    out_dir.mkdir(exist_ok=True)

    print("Preparing data …")
    X_train, y_train, X_test, y_test, feature_cols = prepare()
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}  Features: {len(feature_cols)}")

    print("\nFitting Random Forest + Ridge …")
    rf, ridge = fit_models(X_train, y_train)

    # ----- Random Forest SHAP -----
    print("\n=== Random Forest (TreeExplainer) ===")
    rf_imp, rf_shap, X_explain = shap_random_forest(
        rf, X_test, feature_cols, out_dir
    )

    print("\nTop 15 features by mean |SHAP| (Random Forest):")
    print(rf_imp.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ----- Ridge SHAP -----
    print("\n=== Ridge (LinearExplainer) ===")
    ridge_imp, ridge_coefs = shap_ridge(
        ridge, X_train, X_test, feature_cols, out_dir
    )

    print("\nTop 15 features by mean |SHAP| (Ridge):")
    print(ridge_imp.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nTop 10 Ridge coefficients (signed):")
    print(ridge_coefs.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ----- Combined table -----
    combined = (
        rf_imp.rename(columns={"mean_abs_shap": "rf_mean_abs_shap"})
        .merge(
            ridge_imp.rename(columns={"mean_abs_shap": "ridge_mean_abs_shap"}),
            on="feature",
            how="outer",
        )
        .merge(
            ridge_coefs[["feature", "coefficient"]],
            on="feature",
            how="left",
        )
        .sort_values("rf_mean_abs_shap", ascending=False)
    )
    csv_path = out_dir / "shap_importance.csv"
    combined.to_csv(csv_path, index=False)
    print(f"\nFull importance table → {csv_path}")

    # Quick interpretation
    print("\n=== Quick interpretation ===")
    top5 = rf_imp.head(5)["feature"].tolist()
    print(f"RF top-5 drivers of next-day clear-sky solar: {top5}")
    print(
        "Expect lag_solar_* , roll_mean_* and Fourier (doy_sin / doy_cos) "
        "to dominate – clear-sky irradiance is highly persistent and seasonal."
    )


if __name__ == "__main__":
    main()
