# backend/debug_csv.py
from pathlib import Path
import pandas as pd
import os, sys

HERE = Path(__file__).parent.resolve()
cand1 = HERE / "data" / "resources_clean.csv"
cand2 = HERE / "data" / "resources.csv"
print("HERE:", HERE)
print("cand1 exists:", cand1.exists(), "path:", cand1)
print("cand2 exists:", cand2.exists(), "path:", cand2)
p = cand1 if cand1.exists() else (cand2 if cand2.exists() else None)
print("Using path:", p)

if p:
    try:
        df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8")
    except Exception as e:
        print("read_csv failed (fast):", e)
        df = pd.read_csv(p, dtype=str, on_bad_lines="skip", encoding="utf-8")
    print("rows total:", len(df))
    print("columns:", df.columns.tolist())
    print("first 5 rows (as dicts):")
    print(df.head(5).to_dict(orient="records"))
else:
    print("No CSV file found at expected paths.")
