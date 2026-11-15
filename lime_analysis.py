"""
Klima – LIME Feature Attribution Analysis
=========================================
Local Interpretable Model-agnostic Explanations for next-day clear-sky
solar forecasts (Random Forest + Ridge).

Usage
-----
    cd klima_v2
    python src/lime_analysis.py

Outputs
-------
- Console: local explanations for several representative test days
- models/lime_rf_example.png      – bar chart for one RF explanation
- models/lime_ridge_example.png   – bar chart for one Ridge explanation
- models/lime_global_approx.csv   – average absolute LIME weights (pseudo-global)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import load_data, make_features


# ---------------------------------------------------------------------------
# Data + models (same chronological setup as the rest of Klima)
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
    dates = test["date"].values

    return X_train, y_train, X_test, y_test, feature_cols, dates


def fit_models(X_train, y_train):
    rf = RandomForestRegressor(
        n_estimators=300,
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
# LIME helpers
# ---------------------------------------------------------------------------
def make_explainer(X_train: pd.DataFrame, feature_cols: list[str]) -> LimeTabularExplainer:
    """
    Continuous features only (our engineered set has no true categoricals).
    mode='regression' is required for continuous targets.
    """
    return LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_cols,
        mode="regression",
        discretize_continuous=True,   # LIME’s usual binning for readability
        random_state=42,
    )


def explain_instance(
    explainer: LimeTabularExplainer,
    model,
    x: np.ndarray,
    feature_cols: list[str],
    num_features: int = 12,
):
    """Return (prediction, list of (feature, weight) pairs)."""
    # LIME expects a 2-d predict_fn
    def predict_fn(data):
        return model.predict(data)

    exp = explainer.explain_instance(
        data_row=x,
        predict_fn=predict_fn,
        num_features=num_features,
        num_samples=2000,          # local neighbourhood size
    )
    pred = float(model.predict(x.reshape(1, -1))[0])
    # exp.as_list() → [(feature_desc, weight), ...]
    return pred, exp.as_list(), exp


def plot_explanation(exp, title: str, path: Path):
    """Save a horizontal bar chart of the local LIME weights."""
    items = exp.as_list()
    labels = [t[0] for t in items][::-1]
    weights = [t[1] for t in items][::-1]
    colors = ["#16a34a" if w > 0 else "#dc2626" for w in weights]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(labels, weights, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LIME weight (local linear contribution)")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def aggregate_lime(
    explainer,
    model,
    X_test: pd.DataFrame,
    feature_cols: list[str],
    n_instances: int = 40,
) -> pd.DataFrame:
    """
    Approximate global importance by averaging absolute LIME weights
    over a random subset of test instances.
    """
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_test), min(n_instances, len(X_test)), replace=False)

    # Map from clean feature name → list of |weights|
    accum = {f: [] for f in feature_cols}

    for i in idx:
        x = X_test.iloc[i].values
        _, pairs, _ = explain_instance(explainer, model, x, feature_cols, num_features=15)
        # LIME returns strings like "roll_mean_3 <= 26.40"; extract base name
        for desc, w in pairs:
            base = desc.split(" ")[0].split("<")[0].split(">")[0].strip()
            # fuzzy match against known features
            matched = None
            for f in feature_cols:
                if f in base or base in f:
                    matched = f
                    break
            if matched is None:
                # try exact token
                for f in feature_cols:
                    if f == base:
                        matched = f
                        break
            if matched:
                accum[matched].append(abs(w))

    rows = []
    for f, vals in accum.items():
        rows.append({
            "feature": f,
            "mean_abs_lime": float(np.mean(vals)) if vals else 0.0,
            "n_appearances": len(vals),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("mean_abs_lime", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    out_dir = Path(__file__).resolve().parents[1] / "models"
    out_dir.mkdir(exist_ok=True)

    print("Preparing data …")
    X_train, y_train, X_test, y_test, feature_cols, dates = prepare()
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}  Features: {len(feature_cols)}")

    print("Fitting models …")
    rf, ridge = fit_models(X_train, y_train)

    print("Building LIME explainer (on training distribution) …")
    explainer = make_explainer(X_train, feature_cols)

    # ------------------------------------------------------------------
    # 1. Local explanations for a few interesting test days
    # ------------------------------------------------------------------
    # Pick: a high-solar day, a low-solar day, and a median day
    y_vals = y_test.values
    high_i = int(np.argmax(y_vals))
    low_i = int(np.argmin(y_vals))
    med_i = int(np.argsort(y_vals)[len(y_vals) // 2])

    cases = [
        ("High solar day", high_i),
        ("Low solar day", low_i),
        ("Median solar day", med_i),
    ]

    print("\n=== Local LIME explanations (Random Forest) ===")
    for label, i in cases:
        x = X_test.iloc[i].values
        true = y_test.iloc[i]
        pred, pairs, exp = explain_instance(explainer, rf, x, feature_cols)
        date_str = pd.Timestamp(dates[i]).strftime("%Y-%m-%d")
        print(f"\n--- {label}  ({date_str}) ---")
        print(f"  Actual next-day solar : {true:.2f} MJ/m²")
        print(f"  RF prediction         : {pred:.2f} MJ/m²")
        print("  Top local contributions:")
        for feat, w in pairs[:8]:
            direction = "↑" if w > 0 else "↓"
            print(f"    {direction} {feat:<45} {w:+.3f}")

        if label == "Median solar day":
            plot_explanation(
                exp,
                title=f"LIME – RF – {date_str} (median day)",
                path=out_dir / "lime_rf_example.png",
            )

    print("\n=== Local LIME explanations (Ridge) ===")
    for label, i in cases:
        x = X_test.iloc[i].values
        true = y_test.iloc[i]
        pred, pairs, exp = explain_instance(explainer, ridge, x, feature_cols)
        date_str = pd.Timestamp(dates[i]).strftime("%Y-%m-%d")
        print(f"\n--- {label}  ({date_str}) ---")
        print(f"  Actual next-day solar : {true:.2f} MJ/m²")
        print(f"  Ridge prediction      : {pred:.2f} MJ/m²")
        print("  Top local contributions:")
        for feat, w in pairs[:8]:
            direction = "↑" if w > 0 else "↓"
            print(f"    {direction} {feat:<45} {w:+.3f}")

        if label == "Median solar day":
            plot_explanation(
                exp,
                title=f"LIME – Ridge – {date_str} (median day)",
                path=out_dir / "lime_ridge_example.png",
            )

    # ------------------------------------------------------------------
    # 2. Approximate global ranking via average |LIME weight|
    # ------------------------------------------------------------------
    print("\n=== Approximate global importance (avg |LIME| over 40 test points) ===")
    print("Random Forest …")
    rf_global = aggregate_lime(explainer, rf, X_test, feature_cols, n_instances=40)
    print(rf_global.head(12).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nRidge …")
    ridge_global = aggregate_lime(explainer, ridge, X_test, feature_cols, n_instances=40)
    print(ridge_global.head(12).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Combined CSV
    combined = (
        rf_global.rename(columns={"mean_abs_lime": "rf_mean_abs_lime", "n_appearances": "rf_n"})
        .merge(
            ridge_global.rename(
                columns={"mean_abs_lime": "ridge_mean_abs_lime", "n_appearances": "ridge_n"}
            ),
            on="feature",
            how="outer",
        )
        .sort_values("rf_mean_abs_lime", ascending=False)
    )
    csv_path = out_dir / "lime_global_approx.csv"
    combined.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    print("\n=== Notes ===")
    print("• LIME explains *individual predictions* by fitting a sparse linear model")
    print("  in a local neighbourhood of the instance.")
    print("• Positive weight → feature pushes the forecast higher; negative → lower.")
    print("• Global ranking is only an approximation (average of local |weights|).")
    print("• Compare with SHAP (TreeExplainer / LinearExplainer) for consistency.")


if __name__ == "__main__":
    main()
