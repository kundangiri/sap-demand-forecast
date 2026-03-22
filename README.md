# SAP Intelligent Demand Forecasting
### ERP + Data Science | XGBoost · SHAP · Streamlit

> Bridging SAP SD/MM module data structures with machine learning to predict material demand — built by an SAP Technical Consultant completing an MSc in Data Science.

---

## Why this project?

Most demand forecasting projects on GitHub use generic retail CSVs. This one is different: the data is modelled after **real SAP table structures** (VBAK, VBAP, MARA, KNA1, T001W), the feature engineering reflects how SAP organises transactional data, and the output dashboard mimics **SAP Fiori design language**.

This is what an ERP + Data Science hybrid role actually looks like in practice.

---

## SAP Data Context

| SAP Table | Module | Description | Rows (synthetic) |
|-----------|--------|-------------|-----------------|
| `VBAK`    | SD     | Sales Order Header (order date, customer, sales org) | 6,000 |
| `VBAP`    | SD     | Sales Order Item (material, qty, price, plant) | ~15,000 |
| `MARA`    | MM     | General Material Data (material group, type, UoM) | 40 |
| `KNA1`    | SD     | Customer Master (name, country, customer class) | 80 |
| `T001W`   | MM     | Plant / Storage Location master | 5 |

> **Note:** In a real SAP environment, this data would be extracted via RFC calls, SAP Data Services, or directly from the underlying HANA/Oracle tables using ABAP reports or CDS views.

---

## Project Structure

```
sap_demand_forecast/
├── data/
│   ├── generate_sap_data.py     # Synthetic SAP table generator
│   ├── VBAK.csv                 # Sales order headers
│   ├── VBAP.csv                 # Sales order items
│   ├── MARA.csv                 # Material master
│   ├── KNA1.csv                 # Customer master
│   ├── T001W.csv                # Plant master
│   └── monthly_demand_features.csv  # Engineered feature set
├── src/
│   ├── train_model.py           # Feature engineering + XGBoost training
│   ├── xgb_model.pkl            # Trained model artifact
│   └── shap_explainer.pkl       # SHAP explainer artifact
├── dashboard/
│   └── app.py                   # Streamlit dashboard (SAP Fiori style)
└── README.md
```

---

## Methodology

### 1. Data Generation
Synthetic data replicates SAP table schemas with realistic business patterns:
- Seasonal demand spikes in Q4 (October–December)
- Summer demand dip (June–July)
- Multiple sales organisations, distribution channels, and material groups

### 2. Feature Engineering
Features are derived exactly as they would be from SAP SD extracts:

| Feature | SAP Source | Description |
|---------|-----------|-------------|
| `LAG_1..12` | VBAP.KWMENG | Lagged monthly demand |
| `ROLL_3_MEAN` | VBAP.KWMENG | 3-month rolling average |
| `IS_Q4` | VBAK.ERDAT | Q4 seasonality flag |
| `MATKL_ENC` | MARA.MATKL | Encoded material group |

### 3. Model
- **XGBoost Regressor** — handles non-linear patterns and seasonal effects
- **Train/test split**: 80/20, no shuffle (respects time ordering)
- **SHAP**: TreeExplainer for feature attribution per prediction

### 4. Dashboard
Streamlit app with SAP Fiori-inspired design:
- KPI tiles (total demand, avg monthly, model MAPE)
- Actual vs forecast chart with confidence zone
- SHAP feature importance bar chart
- Demand heatmap by material group and year
- Forecast export table (SAP-style period format YYYYMM)

---

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy faker scikit-learn xgboost streamlit shap plotly

# 2. Generate synthetic SAP data
python data/generate_sap_data.py

# 3. Train the model
python src/train_model.py

# 4. Launch the dashboard
streamlit run dashboard/app.py
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| pandas | Data manipulation (SAP table joins) |
| XGBoost | Demand forecasting model |
| SHAP | Model explainability |
| Streamlit | Interactive dashboard |
| Plotly | Charts |
| Faker | Realistic synthetic data |

---

## Skills Demonstrated

- **SAP domain knowledge**: SD/MM module table structures, field naming conventions (VBELN, MATNR, KWMENG, ERDAT), sales org hierarchy
- **Data engineering**: Multi-table joins mimicking SAP CDS views, lag/rolling feature construction on time-series ERP data
- **Machine learning**: XGBoost with temporal train/test split, hyperparameter tuning, MAPE/RMSE evaluation
- **Explainability**: SHAP TreeExplainer for business-friendly model interpretation
- **Visualisation**: SAP Fiori-style dashboard design in Streamlit

---

## Author

MSc Data Science candidate with 2+ years of SAP Technical Consulting experience (SD/MM modules).  
Bridging enterprise ERP systems with modern data science — the gap most companies struggle to fill.

---

*This project uses synthetic data. In production, data would be extracted from SAP via RFC, BAPIs, SAP Data Services, or HANA SQL.*
