"""
Klima v2 – Pipeline: multi-year NASA POWER data → features → chronological split
→ Ridge + RandomForest + TabPFN → metrics + battery decision layer
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Optional TabPFN
try:
    from tabpfn import TabPFNRegressor
    HAS_TABPFN = True
except Exception:
    HAS_TABPFN = False
    TabPFNRegressor = None


def _parse_orig(path: Path) -> pd.DataFrame:
    """Original 2026 file (MJ units, YEAR+DOY)."""
    df = pd.read_csv(path, skiprows=14)
    df = df.replace(-999.0, np.nan)
    df["date"] = pd.to_datetime(
        df["YEAR"].astype(str) + df["DOY"].astype(str).str.zfill(3), format="%Y%j"
    )
    df = df.rename(
        columns={
            "CLRSKY_SFC_SW_DWN": "solar",
            "CLRSKY_SFC_PAR_TOT": "par",
        }
    )
    df["allsky"] = np.nan
    keep = ["date", "solar", "par", "T2M", "TS", "RH2M", "PS", "allsky"]
    return df[[c for c in keep if c in df.columns]]


def _parse_hist(path: Path) -> pd.DataFrame:
    """Historical files from API (kWh → MJ * 3.6)."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(
        df[["YEAR", "MO", "DY"]].astype(str).agg("-".join, axis=1)
    )
    df["solar"] = df["CLRSKY_SFC_SW_DWN"] * 3.6
    df["par"] = df.get("CLRSKY_SFC_PAR_TOT", np.nan) * 3.6
    df["allsky"] = df.get("ALLSKY_SFC_SW_DWN", np.nan) * 3.6
    keep = ["date", "solar", "par", "T2M", "TS", "RH2M", "PS", "allsky"]
    return df[[c for c in keep if c in df.columns]]


def load_data(data_dir: Path) -> pd.DataFrame:
    """
    Load every available NASA POWER CSV in data/, convert units consistently
    to MJ/m²/day, keep all rows that have a valid solar value.
    """
    data_dir = Path(data_dir)
    frames = []

    # Original provided file
    orig = list(data_dir.glob("POWER_Point_Daily_*.csv"))
    for p in orig:
        frames.append(_parse_orig(p))

    # Extra historical downloads
    for p in sorted(data_dir.glob("nasa_*.csv")):
        frames.append(_parse_hist(p))

    if not frames:
        raise FileNotFoundError(f"No usable CSV found in {data_dir}")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["solar"]).reset_index(drop=True)

    # Ensure numeric
    for c in ["solar", "par", "T2M", "TS", "RH2M", "PS", "allsky"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag / rolling / calendar features for next-day solar prediction."""
    df = df.copy()
    df["target"] = df["solar"].shift(-1)  # next-day clear-sky irradiance

    # Lags of target-related series
    for lag in [1, 2, 3, 7, 14]:
        df[f"lag_solar_{lag}"] = df["solar"].shift(lag)
        if "T2M" in df.columns:
            df[f"lag_t2m_{lag}"] = df["T2M"].shift(lag)
        if "RH2M" in df.columns:
            df[f"lag_rh_{lag}"] = df["RH2M"].shift(lag)
        if "PS" in df.columns:
            df[f"lag_ps_{lag}"] = df["PS"].shift(lag)

    # Rolling statistics
    df["roll_mean_3"] = df["solar"].rolling(3, min_periods=1).mean()
    df["roll_mean_7"] = df["solar"].rolling(7, min_periods=1).mean()
    df["roll_mean_14"] = df["solar"].rolling(14, min_periods=1).mean()
    df["roll_std_7"] = df["solar"].rolling(7, min_periods=1).std()
    df["roll_std_14"] = df["solar"].rolling(14, min_periods=1).std()
    df["roll_min_7"] = df["solar"].rolling(7, min_periods=1).min()
    df["roll_max_7"] = df["solar"].rolling(7, min_periods=1).max()

    # Calendar + Fourier seasonality (critical for clear-sky)
    df["doy"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    df["doy_sin2"] = np.sin(4 * np.pi * df["doy"] / 365.25)
    df["doy_cos2"] = np.cos(4 * np.pi * df["doy"] / 365.25)

    # Simple interaction
    if "T2M" in df.columns and "RH2M" in df.columns:
        df["t2m_x_rh"] = df["T2M"] * df["RH2M"] / 100.0

    # Drop rows that lost target or needed lags
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    # Fill remaining NaNs in features with column median (rare)
    feat_cols = [c for c in df.columns if c not in ("date", "target", "solar", "par", "allsky", "TS")]
    for c in feat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    return df


def train_and_evaluate(df: pd.DataFrame, train_ratio: float = 0.85):
    """Chronological train/test split, fit three models, return metrics + preds."""
    exclude = {
        "YEAR", "DOY", "MO", "DY", "date", "target", "solar",
        "par", "allsky", "TS", "CLRSKY_SFC_SW_DWN", "CLRSKY_SFC_PAR_TOT",
        "ALLSKY_SFC_SW_DWN",
    }
    feature_cols = [c for c in df.columns if c not in exclude and not c.startswith("Unnamed")]

    split = int(len(df) * train_ratio)
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()

    X_train = train[feature_cols].astype(np.float64)
    y_train = train["target"].astype(np.float64)
    X_test = test[feature_cols].astype(np.float64)
    y_test = test["target"].astype(np.float64)
    y_today = test["solar"].values

    # --- Ridge ---
    lin = make_pipeline(StandardScaler(), Ridge(alpha=0.5))
    lin.fit(X_train, y_train)
    pred_lin = lin.predict(X_test)

    # --- Random Forest ---
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    # --- TabPFN ---
    pred_tab = None
    tab_model = None
    if HAS_TABPFN:
        try:
            # TabPFN prefers modest sample counts; subsample if needed
            max_train = 2500
            if len(X_train) > max_train:
                idx = np.linspace(0, len(X_train) - 1, max_train, dtype=int)
                X_tab = X_train.iloc[idx]
                y_tab = y_train.iloc[idx]
            else:
                X_tab, y_tab = X_train, y_train

            tab_model = TabPFNRegressor(device="cpu", n_estimators=8)
            tab_model.fit(X_tab, y_tab)
            pred_tab = tab_model.predict(X_test)
        except Exception as e:
            print(f"TabPFN failed: {e}")
            pred_tab = pred_rf.copy()  # fallback
    else:
        pred_tab = pred_rf.copy()

    # Persistence baseline
    pred_pers = y_today

    def metrics(y, p):
        return {
            "r2": float(r2_score(y, p)),
            "mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
        }

    results = {
        "feature_cols": feature_cols,
        "models": {
            "linear": lin,
            "random_forest": rf,
            "tabpfn": tab_model,
        },
        "metrics": {
            "persistence": metrics(y_test, pred_pers),
            "linear": metrics(y_test, pred_lin),
            "random_forest": metrics(y_test, pred_rf),
            "tabpfn": metrics(y_test, pred_tab),
        },
        "test": {
            "dates": test["date"].values,
            "y_true": y_test.values,
            "y_today": y_today,
            "pred_lin": pred_lin,
            "pred_rf": pred_rf,
            "pred_tab": pred_tab,
            "pred_pers": pred_pers,
        },
        "train_size": len(train),
        "test_size": len(test),
        "latest_row": test.iloc[-1].to_dict(),
        "has_tabpfn": HAS_TABPFN and tab_model is not None,
    }
    return results


def decision(predicted: float, threshold: float = 26.0) -> dict:
    """Battery decision rule (values in MJ/m²/day)."""
    good = predicted > threshold
    return {
        "is_good": good,
        "action": "Leave room for the sun" if good else "Charge from grid",
        "detail": (
            "Keep backup reserve low and leave capacity for solar charging tomorrow."
            if good
            else "Low solar expected. Fully charge the battery from the grid overnight."
        ),
        "reserve_pct": 20 if good else 75,
        "threshold": threshold,
    }


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[1] / "data"
    df = load_data(data_dir)
    print("Rows after clean:", len(df))
    print("Date range:", df["date"].min(), "→", df["date"].max())
    df = make_features(df)
    print("Rows after features:", len(df))
    res = train_and_evaluate(df, train_ratio=0.85)
    print("Train/Test:", res["train_size"], res["test_size"])
    for name, m in res["metrics"].items():
        print(f"{name:15s}  R²={m['r2']:.4f}  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}")
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(exist_ok=True)
    # Do not pickle TabPFN (large / not always picklable); store metrics + sklearn models
    save_res = {k: v for k, v in res.items() if k != "models"}
    save_res["models"] = {
        "linear": res["models"]["linear"],
        "random_forest": res["models"]["random_forest"],
    }
    joblib.dump(save_res, models_dir / "results.joblib")
    print("Saved results.joblib")
