from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "early_snapshot_ui.parquet"
OUT = ROOT / "models" / "ui_model.joblib"

FEATURES = [
    "util_proxy",
    "delinq_proxy",
    "spend_vol_proxy",
    "income_stab_proxy",
    "util_trend",
    "spend_vol_avg",
    "missed_rate",
    "history_len",
]

def main() -> None:
    df = pd.read_parquet(DATA)

    X = df[FEATURES].copy()
    y = df["target"].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.08,
            max_iter=250,
            random_state=42
        )),
    ])

    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p)
    print("AUC:", round(auc, 4))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, OUT)
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
