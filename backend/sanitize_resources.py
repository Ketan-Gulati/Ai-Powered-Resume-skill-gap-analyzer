# sanitize_resources.py  (fixed)
import io, csv, os, sys, codecs

SRC = os.path.join("data","resources.csv")
OUT = os.path.join("data","resources_clean.csv")

if not os.path.exists(SRC):
    print("Source not found:", SRC); sys.exit(1)

def normalize_text(s):
    if s is None: return ""
    s = s.replace('\u2018', "'").replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
    s = s.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
    return s.strip()

bad = total = 0
# use builtin open with newline='' (csv writer needs this)
with open(SRC, "r", encoding="utf-8", errors="replace") as inf, \
     open(OUT, "w", encoding="utf-8", errors="replace", newline='') as outf:
    writer = csv.writer(outf, quoting=csv.QUOTE_MINIMAL)
    for raw in inf:
        total += 1
        line = raw.rstrip("\n\r")
        if len(line) >= 2 and line[0] == "'" and line[-1] == "'":
            line = line[1:-1]
        parts = line.split("\t")
        parts = [normalize_text(p) for p in parts]
        if len(parts) <= 1 and all(not p for p in parts):
            bad += 1
            continue
        try:
            writer.writerow(parts)
        except Exception as e:
            bad += 1
            print("WRITE ERROR line", total, "->", e)
            continue

print(f"Done. Wrote {OUT} (input lines: {total}, skipped: {bad})")