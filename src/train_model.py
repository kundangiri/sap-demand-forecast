"""
SAP Demand Forecasting - Feature Engineering & Model Training
Uses VBAK + VBAP tables to build monthly demand forecasts per material.
Models: Prophet (baseline) + XGBoost (main) + SHAP explainability
"""

import pandas as pd
import numpy as np
import pickle, os, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from xgboost import XGBRegressor
import shap

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "src")

# ── 1. Load SAP tables ───────────────────────────────────────────────────
def load_tables():
    vbak  = pd.read_csv(f"{DATA_DIR}/VBAK.csv", parse_dates=["ERDAT"])
    vbap  = pd.read_csv(f"{DATA_DIR}/VBAP.csv", parse_dates=["ERDAT"])
    mara  = pd.read_csv(f"{DATA_DIR}/MARA.csv")
    kna1  = pd.read_csv(f"{DATA_DIR}/KNA1.csv")
    return vbak, vbap, mara, kna1

# ── 2. Build monthly demand time-series ─────────────────────────────────
def build_monthly_demand(vbak, vbap, mara):
    vbap = vbap.drop(columns=["MATKL"], errors="ignore")
    df = vbap.merge(vbak[["VBELN","ERDAT","AUART","VKORG"]].rename(columns={"ERDAT":"ERDAT_HDR"}), on="VBELN")
    df = df.merge(mara[["MATNR","MATKL","MTART"]], on="MATNR")

    # Filter: only standard orders (not returns)
    df = df[df["AUART"] != "RE"]

    df["YEAR_MONTH"] = df["ERDAT"].dt.to_period("M")
    monthly = (
        df.groupby(["MATNR","MATKL","YEAR_MONTH"])
          .agg(DEMAND=("KWMENG","sum"), REVENUE=("NETWR","sum"), N_ORDERS=("VBELN","nunique"))
          .reset_index()
    )
    monthly["YEAR_MONTH"] = monthly["YEAR_MONTH"].dt.to_timestamp()
    monthly = monthly.sort_values(["MATNR","YEAR_MONTH"]).reset_index(drop=True)
    return monthly

# ── 3. Feature engineering ───────────────────────────────────────────────
def engineer_features(monthly):
    df = monthly.copy()
    df = df.sort_values(["MATNR","YEAR_MONTH"])

    # Time features
    df["MONTH"]   = df["YEAR_MONTH"].dt.month
    df["QUARTER"] = df["YEAR_MONTH"].dt.quarter
    df["YEAR"]    = df["YEAR_MONTH"].dt.year

    # Seasonality flags
    df["IS_Q4"]     = (df["QUARTER"] == 4).astype(int)
    df["IS_SUMMER"] = (df["MONTH"].isin([6, 7, 8])).astype(int)

    # Lag features (per material)
    for lag in [1, 2, 3, 6, 12]:
        df[f"LAG_{lag}"] = df.groupby("MATNR")["DEMAND"].shift(lag)

    # Rolling stats
    df["ROLL_3_MEAN"]  = df.groupby("MATNR")["DEMAND"].transform(lambda x: x.shift(1).rolling(3).mean())
    df["ROLL_6_MEAN"]  = df.groupby("MATNR")["DEMAND"].transform(lambda x: x.shift(1).rolling(6).mean())
    df["ROLL_3_STD"]   = df.groupby("MATNR")["DEMAND"].transform(lambda x: x.shift(1).rolling(3).std())

    # Material group encoding
    df["MATKL_ENC"] = pd.factorize(df["MATKL"])[0]

    df = df.dropna()
    return df

# ── 4. Train XGBoost ─────────────────────────────────────────────────────
FEATURES = [
    "MONTH","QUARTER","YEAR","IS_Q4","IS_SUMMER","MATKL_ENC",
    "LAG_1","LAG_2","LAG_3","LAG_6","LAG_12",
    "ROLL_3_MEAN","ROLL_6_MEAN","ROLL_3_STD"
]

def train_xgb(df):
    X = df[FEATURES]
    y = df["DEMAND"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return model, X_train, X_test, y_train, y_test, y_pred, mape, rmse

# ── 5. SHAP explainability ───────────────────────────────────────────────
def compute_shap(model, X_test):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values

# ── 6. Save artifacts ────────────────────────────────────────────────────
def save_artifacts(model, explainer, monthly_feat, mape, rmse):
    with open(f"{MODEL_DIR}/xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{MODEL_DIR}/shap_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)
    monthly_feat.to_csv(f"{DATA_DIR}/monthly_demand_features.csv", index=False)
    metrics = {"MAPE": round(mape, 4), "RMSE": round(rmse, 2)}
    pd.DataFrame([metrics]).to_csv(f"{DATA_DIR}/model_metrics.csv", index=False)
    print(f"  Model MAPE : {mape:.2%}")
    print(f"  Model RMSE : {rmse:.2f}")
    return metrics

# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading SAP tables...")
    vbak, vbap, mara, kna1 = load_tables()

    print("Building monthly demand series...")
    monthly = build_monthly_demand(vbak, vbap, mara)
    print(f"  {len(monthly):,} material-month records")

    print("Engineering features...")
    df_feat = engineer_features(monthly)
    print(f"  {len(df_feat):,} rows after lag/rolling features")

    print("Training XGBoost model...")
    model, X_tr, X_te, y_tr, y_te, y_pred, mape, rmse = train_xgb(df_feat)

    print("Computing SHAP values...")
    explainer, shap_values = compute_shap(model, X_te)

    print("Saving artifacts...")
    metrics = save_artifacts(model, explainer, df_feat, mape, rmse)

    print("\nTraining complete.")
