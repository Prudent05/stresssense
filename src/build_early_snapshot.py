from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Adjust these if your files differ
PARTS_GLOB = "train_sample_part_*.parquet"  # your sampled parts
LABELS_CSV = DATA_DIR / "train_labels.csv"
OUT_PATH = DATA_DIR / "early_snapshot_ui.parquet"

# "Early" = 3rd-from-last statement (~2 months before latest statement)
EARLY_OFFSET_FROM_END = 3


def build_ui_features(group: pd.DataFrame) -> dict:
    """
    group: all rows for one customer, sorted by statement date asc.
    We create consumer-friendly features using ONLY data up to the early prediction point.
    """
    group = group.sort_values("S_2")  # AmEx statement date column is S_2
    if len(group) < EARLY_OFFSET_FROM_END:
        return {}  # skip too-short histories for MVP

    pred_row = group.iloc[-EARLY_OFFSET_FROM_END]        # prediction statement row
    history = group.iloc[: len(group) - EARLY_OFFSET_FROM_END + 1]  # <= prediction point

    # Consumer proxies (these are "signals", not perfect truths)
    # We keep them numeric and stable.
    util_proxy = float(pd.to_numeric(pred_row.get("B_1"), errors="coerce") or np.nan)
    delin_proxy = float(pd.to_numeric(pred_row.get("D_39"), errors="coerce") or np.nan)
    spend_vol_proxy = float(pd.to_numeric(pred_row.get("S_3"), errors="coerce") or np.nan)
    income_stab_proxy = float(pd.to_numeric(pred_row.get("P_2"), errors="coerce") or np.nan)

    # Behavior trend features computed from HISTORY ONLY (leakage-safe)
    # (If column missing, we fall back to NaN; training will impute)
    def safe_series(col: str) -> pd.Series:
        if col not in history.columns:
            return pd.Series(dtype="float64")
        return pd.to_numeric(history[col], errors="coerce")

    b1_hist = safe_series("B_1")
    s3_hist = safe_series("S_3")
    d39_hist = safe_series("D_39")

    util_trend = float((b1_hist.iloc[-1] - b1_hist.iloc[0]) if len(b1_hist.dropna()) >= 2 else np.nan)
    spend_vol_avg = float(s3_hist.mean()) if len(s3_hist.dropna()) else np.nan
    missed_rate = float((d39_hist > 0).mean()) if len(d39_hist.dropna()) else np.nan

    return {
        "customer_ID": pred_row["customer_ID"],
        "util_proxy": util_proxy,
        "delinq_proxy": delin_proxy,
        "spend_vol_proxy": spend_vol_proxy,
        "income_stab_proxy": income_stab_proxy,
        "util_trend": util_trend,
        "spend_vol_avg": spend_vol_avg,
        "missed_rate": missed_rate,
        "history_len": int(len(history)),
    }


def main() -> None:
    parts = sorted(DATA_DIR.glob(PARTS_GLOB))
    if not parts:
        raise FileNotFoundError(f"No parquet parts found in {DATA_DIR} matching {PARTS_GLOB}")

    labels = pd.read_csv(LABELS_CSV)
    labels = labels[["customer_ID", "target"]].copy()

    rows = []
    for i, p in enumerate(parts, start=1):
        df = pd.read_parquet(p)

        # Require these columns to exist
        needed = {"customer_ID", "S_2"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"{p.name} is missing required columns: {missing}")

        for cid, grp in df.groupby("customer_ID", sort=False):
            feat = build_ui_features(grp)
            if feat:
                rows.append(feat)

        if i % 5 == 0:
            print(f"Processed {i}/{len(parts)} parts; rows so far: {len(rows):,}")

    snap = pd.DataFrame(rows)
    snap = snap.merge(labels, on="customer_ID", how="inner")

    snap.to_parquet(OUT_PATH, index=False)
    print("Saved:", OUT_PATH)
    print("Shape:", snap.shape)
    print("Target mean:", snap["target"].mean())


if __name__ == "__main__":
    main()
