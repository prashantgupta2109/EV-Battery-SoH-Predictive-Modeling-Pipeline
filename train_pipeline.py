# =============================================================================
#  train_pipeline.py
#  EV Battery State-of-Health (SoH) Predictive Modeling Pipeline
#  Trains a LightGBM regressor on cycle-level battery data.
#
#  Reads from: <script_dir>/data_battery_cycle/
#  Saves to:   <script_dir>/model/
#
#  Run from ANY directory:
#      python path/to/train_pipeline.py
# =============================================================================

import os
import sys
import glob
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# ---------------------------------------------------------------------------
# Resolve ALL paths relative to THIS script's directory, not the shell CWD.
# This guarantees the script works regardless of where it is called from.
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data_battery_cycle")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Features the model is trained on and the target column
FEATURES = ["voltage", "temperature", "cycle"]
TARGET   = "soh"


# ---------------------------------------------------------------------------
# Step 1 - Locate & load any CSV inside data_battery_cycle/
# ---------------------------------------------------------------------------
def find_and_load_csv() -> pd.DataFrame:
    """Finds and loads the first CSV file found inside data_battery_cycle/."""
    pattern   = os.path.join(DATA_DIR, "*.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        os.makedirs(DATA_DIR, exist_ok=True)
        raise FileNotFoundError(
            "\n[ERROR] No CSV file found inside:\n"
            f"  {DATA_DIR}\n\n"
            "Please place your battery cycle dataset (CSV) there and re-run.\n"
        )

    csv_path = csv_files[0]
    print(f"[INFO] Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Dataset loaded - {len(df):,} rows x {df.shape[1]} columns.")
    return df


# ---------------------------------------------------------------------------
# Step 2 - Clean & prepare the DataFrame
# ---------------------------------------------------------------------------
def clean_and_prepare(df: pd.DataFrame):
    """
    Standardises column names, validates required columns, drops NaN rows,
    and returns feature DataFrame X and target Series y.
    Returns X as a DataFrame (not numpy) so LightGBM retains feature names
    and avoids sklearn UserWarning during inference.
    """
    # Normalise column names to lowercase with no surrounding whitespace
    df.columns = [c.lower().strip() for c in df.columns]

    # Validate that every required column exists
    missing = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing:
        raise KeyError(
            "\n[ERROR] The following required columns are missing from the CSV:\n"
            f"  {missing}\n"
            f"Available columns: {list(df.columns)}\n"
        )

    # Drop rows that have NaN in any required column
    required_cols = FEATURES + [TARGET]
    before = len(df)
    df = df.dropna(subset=required_cols).copy()
    after = len(df)
    if before != after:
        print(f"[INFO] Dropped {before - after} rows with missing values "
              f"({after:,} rows remaining).")

    # Clip SoH to [0, 1.2] for robustness (some batteries briefly exceed 1.0)
    df[TARGET] = df[TARGET].clip(0.0, 1.2)

    # Keep X as a named DataFrame so model.predict() receives feature names
    X = df[FEATURES]
    y = df[TARGET]
    return X, y


# ---------------------------------------------------------------------------
# Step 3 - Train, evaluate, save
# ---------------------------------------------------------------------------
def run_pipeline():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Ingest
    df = find_and_load_csv()

    # 2. Process
    X, y = clean_and_prepare(df)

    # 3. Split & scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    # Fit & transform preserving column names (returns numpy, used only inside training)
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURES
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURES
    )

    # 4. Train LightGBM
    print("[INFO] Training LightGBM Regressor ...")
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,         # suppress LightGBM's internal console output
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress sklearn feature-name warnings
        model.fit(
            X_train_sc, y_train,
            eval_set=[(X_test_sc, y_test)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=50),
            ],
        )

    # 5. Evaluate
    preds = model.predict(X_test_sc)
    r2    = r2_score(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))

    print()
    print("=" * 50)
    print("  [OK] MODEL TRAINING COMPLETE")
    print("=" * 50)
    print(f"  Python version : {sys.version.split()[0]}")
    print(f"  Features used  : {FEATURES}")
    print(f"  Target         : {TARGET}")
    print(f"  Training rows  : {len(X_train):,}")
    print(f"  Test rows      : {len(X_test):,}")
    print(f"  R2 Score       : {r2:.6f}  ({r2 * 100:.3f}%)")
    print(f"  RMSE           : {rmse:.6f}")
    print("=" * 50)

    # 6. Persist artifacts using absolute paths
    model_path   = os.path.join(MODEL_DIR, "battery_model.pkl")
    scaler_path  = os.path.join(MODEL_DIR, "scaler.pkl")
    feature_path = os.path.join(MODEL_DIR, "feature_names.pkl")
    metrics_path = os.path.join(MODEL_DIR, "metrics.pkl")

    joblib.dump(model,                      model_path)
    joblib.dump(scaler,                     scaler_path)
    joblib.dump(FEATURES,                   feature_path)
    joblib.dump({"r2": r2, "rmse": rmse},   metrics_path)

    print(f"\n[INFO] Artifacts saved to: {MODEL_DIR}")
    print(f"  - {os.path.basename(model_path)}")
    print(f"  - {os.path.basename(scaler_path)}")
    print(f"  - {os.path.basename(feature_path)}")
    print(f"  - {os.path.basename(metrics_path)}")
    print(f"\n[INFO] Launch dashboard: streamlit run {os.path.join(BASE_DIR, 'dashboard.py')}\n")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()