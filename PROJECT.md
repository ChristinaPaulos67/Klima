# Klima — Next-Day Clear-Sky Solar Forecast for Household Battery Decisions

**Veritas AI International High School Hackathon 2026 · Track: AI + Climate**

> Our project helps a household with rooftop solar + battery by taking today’s weather and recent solar readings and giving back tomorrow’s clear-sky solar forecast plus a simple “leave room for the sun” / “charge from grid” decision.

---

## 1. Problem

Households with solar + battery need to decide overnight how much grid power to store.  
If tomorrow will be sunny, they should leave battery capacity free.  
If tomorrow will be poor, they should charge from the grid.  

A reliable next-day irradiance forecast turns that decision into an automated rule.

## 2. Solution

**Klima** is an end-to-end ML system that:

1. Ingests multi-year NASA POWER daily meteorology + clear-sky radiation for a real location (Addis Ababa region, 9.01°N 38.76°E).
2. Engineers lag, rolling and seasonal features.
3. Trains three models (Ridge, Random Forest, TabPFN) under a strict chronological split.
4. Beats a strong persistence baseline (R² ≈ 0.90) with Ridge R² ≈ 0.91.
5. Converts the forecast into a household battery action via a simple threshold rule.
6. Explains every prediction with SHAP and LIME.

## 3. Data (all original rows used)

| Source | Period | Rows kept |
|--------|--------|-----------|
| Original POWER_Point_Daily_…_UTC.csv | 2026-01-01 → 2026-08-31 | **243** (every valid solar row) |
| NASA POWER API (extra) | 2020-01-01 → 2025-12-31 | 2 192 |
| **Total after clean** | 2020 → 2026 | **2 435** |

Units converted to MJ m⁻² day⁻¹ consistently. Missing values handled; no future leakage.

## 4. Models & Performance (chronological 85/15 split)

| Model | R² | MAE (MJ/m²) | Notes |
|-------|-----|-------------|-------|
| **Ridge** | **0.911** | 0.368 | Best point forecast |
| Random Forest | 0.875 | 0.438 | Strong non-linear baseline |
| TabPFN | ≥ 0.85 | – | Foundation model (token optional) |
| Persistence | 0.896 | 0.405 | Required baseline |

All R² values are positive and exceed the 0.85 target.

## 5. Interpretability

- **SHAP** (TreeExplainer + LinearExplainer) → global feature ranking  
- **LIME** → local explanations for high / low / median solar days  

Dominant drivers (both methods): `roll_mean_3`, lag solar values, Fourier seasonality (`doy_sin` / `doy_cos`). Matches the physics of clear-sky irradiance.

## 6. Decision Layer

```
if predicted_solar > 26 MJ/m²:
    action = "Leave room for the sun"   (reserve 20 %)
else:
    action = "Charge from grid"         (reserve 75 %)
```

## 7. How to run

```bash
cd klima_v2
pip install -r requirements.txt
streamlit run app/app.py          # interactive dashboard
python src/pipeline.py            # retrain + metrics
python src/shap_analysis.py       # SHAP plots + CSV
python src/lime_analysis.py       # LIME local + global
python src/tabpfn_sensitivity.py  # hyper-parameter sweep
```

Optional: `export TABPFN_TOKEN=...` for real TabPFN weights.

## 8. Hackathon alignment (AI + Climate)

- Real climate/energy data (NASA POWER)
- Actionable output for people (battery decision)
- Strong baselines + honest metrics
- Interpretability (SHAP + LIME)
- Working demo (Streamlit)
- All original provided rows retained and used

## 9. Team note

Built for the Veritas AI International High School Hackathon 2026.  
Ready for 2-minute demo video and written submission.
