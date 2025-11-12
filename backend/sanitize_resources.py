# backend/sanitize_resources.py
import os, sys, codecs, csv

SRC = os.path.join(os.path.dirname(_file_), "data", "resources.csv")
OUT = os.path.join(os.path.dirname(_file_), "data", "resources_clean.csv")

if not os.path.exists(SRC):
    print("Source not found:", SRC); sys.exit(1)

def normalize_text(s):
    if s is None: return ""
    s = s.replace('\u2018', "'").replace('\u2019', "'").replace('\u201c','"').replace('\u201d','"')
    s = s.replace('\r\n',' ').replace('\n',' ').replace('\r',' ')
    return s.strip()

total=0; skipped=0
with codecs.open(SRC,"r",encoding="utf-8",errors="replace") as inf, \
     codecs.open(OUT,"w",encoding="utf-8",errors="replace") as outf:
    writer = csv.writer(outf, quoting=csv.QUOTE_MINIMAL)
    for raw in inf:
        total+=1
        line = raw.rstrip("\n\r")
        if len(line)>=2 and line[0]=="'" and line[-1]=="'":
            line=line[1:-1]
        parts = line.split("\t")
        parts = [normalize_text(p) for p in parts]
        if len(parts)==0 or all(not p for p in parts):
            skipped+=1
            continue
        try:
            writer.writerow(parts)
        except Exception:
            skipped+=1
print(f"Done. input lines: {total}, skipped: {skipped}, out: {OUT}")