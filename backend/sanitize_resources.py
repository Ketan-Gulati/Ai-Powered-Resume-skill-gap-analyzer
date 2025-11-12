import pandas as pd
import re
import csv

# Input + output paths
input_path = "data/resources.csv"
output_path = "data/resources_clean.csv"

def clean_cell(v):
    """Cleans up a single cell string."""
    if pd.isna(v):
        return ""
    s = str(v)
    s = re.sub(r'[\r\n]+', ' ', s)
    s = re.sub(r'\s{2,}', ' ', s)
    s = s.replace('\"', '"').strip()
    return s

# --- Try loading flexibly ---
try:
    df = pd.read_csv(
        input_path,
        sep=None,           # Let pandas infer separator
        engine="python",    # Needed for messy CSVs
        quoting=csv.QUOTE_MINIMAL,
        on_bad_lines="skip",
        dtype=str
    )
except Exception as e:
    print("⚠️ Flexible read failed, retrying with defaults:", e)
    df = pd.read_csv(input_path, on_bad_lines="skip", dtype=str, encoding="utf-8")

# --- Clean the data ---
df = df.dropna(axis=1, how="all")  # Drop fully empty columns
df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns

for c in df.columns:
    df[c] = df[c].map(clean_cell)

# --- Keep only relevant textual columns ---
keep_cols = [
    c for c in df.columns
    if any(x in c.lower() for x in [
        "title", "url", "desc", "intro", "skills",
        "rating", "duration", "site", "platform", "course"
    ])
]

if keep_cols:
    df = df[keep_cols]

# --- Save cleaned CSV ---
df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8-sig")

print(f"✅ Cleaned CSV saved to {output_path}")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
print("Columns kept:", ", ".join(df.columns))
