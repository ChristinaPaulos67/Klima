# 🌿 Klima – Solar Digital Twin for Addis Ababa

**Veritas AI International High School Hackathon 2026 · AI + Climate**

Next-day clear-sky solar forecast + household battery decision + digital twin map + climate impact analysis.

> Helps a household (or micro-grid) decide whether to leave battery capacity for tomorrow’s sun or charge from the grid overnight.

## Quick start (Mac / Windows / Linux)

```bash
cd klima_v2
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

Open the browser URL Streamlit prints (usually http://localhost:8501).

## What’s inside

| Page | Purpose |
|------|---------|
| **Overview** | Tomorrow’s forecast, battery decision, recent accuracy |
| **Digital Twin Map** | Addis Ababa grid-cell map + elevation context |
| **Performance** | R² / MAE / RMSE for Ridge, RF, TabPFN, Persistence |
| **History** | Forecast vs actual on the chronological test set |
| **Climate Impact** | Advantages, disadvantages & practical impact of acting on the forecast |
| **About** | Project summary for judges |

## Model quality (kept intact)

| Model | R² | MAE |
|-------|-----|-----|
| Ridge | **0.911** | 0.37 |
| Random Forest | 0.875 | 0.44 |
| TabPFN | ≥ 0.85 | – |
| Persistence | 0.896 | 0.41 |

All original NASA POWER rows are used. Chronological split only.

## Extra analysis scripts

```bash
python src/pipeline.py
python src/shap_analysis.py
python src/lime_analysis.py
python src/tabpfn_sensitivity.py
```

## Decision rule

- Predicted solar **> 26 MJ/m²** → “Leave room for the sun”
- Otherwise → “Charge from grid”

Ready for a ≤ 2-minute demo video and written submission.
