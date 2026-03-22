"""
Startup helper — runs automatically when the app loads on Streamlit Cloud.
Generates SAP data and trains the model if artifacts are missing.
"""
import os, sys, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))

def ensure_artifacts():
    model_path = os.path.join(ROOT, "src", "xgb_model.pkl")
    data_path  = os.path.join(ROOT, "data", "monthly_demand_features.csv")

    if not os.path.exists(data_path):
        print("Generating SAP synthetic data...")
        subprocess.run([sys.executable, os.path.join(ROOT, "data", "generate_sap_data.py")], check=True)

    if not os.path.exists(model_path):
        print("Training XGBoost model...")
        subprocess.run([sys.executable, os.path.join(ROOT, "src", "train_model.py")], check=True)

ensure_artifacts()
