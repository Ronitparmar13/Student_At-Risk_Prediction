"""Train the bundled educational demo model from training_reference.csv."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "training_reference.csv"
OUT = ROOT / "student_risk_model.pkl"
TARGET = "At_Risk"

def main():
    df = pd.read_csv(DATA)
    if TARGET not in df:
        raise ValueError(f"Training data must contain {TARGET}")
    X = df.drop(columns=[TARGET]).copy()
    y = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    categorical = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    prep = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                   ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ], remainder="drop")
    model = Pipeline([("preprocess", prep), ("classifier", RandomForestClassifier(
        n_estimators=350, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
    ))])
    model.fit(X, y)
    bundle = {"model": model, "feature_columns": X.columns.tolist(), "target": TARGET,
              "training_rows": int(len(df)), "model_type": "RandomForestClassifier pipeline",
              "positive_class": 1}
    joblib.dump(bundle, OUT)
    print(f"Saved {OUT} using {len(df)} training rows and {len(X.columns)} input features")

if __name__ == "__main__":
    main()
