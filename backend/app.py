# app.py
import os
import sys
import re
import csv
import logging
import time
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer, util

# optional PDF extractors
try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except Exception:
    pdfplumber = None
    _HAS_PDFPLUMBER = False

try:
    import PyPDF2
    _HAS_PYPDF2 = True
except Exception:
    PyPDF2 = None
    _HAS_PYPDF2 = False

# local semantic extractor (assumes this file exists)
import semantic_extractor as semx

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("skill-gap-backend")

app = FastAPI(title="Skill Gap Analyzer (backend)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ---------------- helpers ----------------
STOPWORDS = {
    "and","or","the","a","an","to","for","of","in","on","with","by",
    "is","are","be","as","that","this","we","you","will","your","have","has",
    "using","use","can","should","may","role","responsible","experience","years","equivalent"
}

# additional blacklist for tokens that are not skills
EXTRA_BLACKLIST = {"json"}

TECH_PHRASES = [
    "node js","node.js","react js","reactjs","next js","nextjs","express js","expressjs",
    "mongo db","mongodb","rest api","restful api","machine learning","deep learning",
    "data science","c++","c#","aws","cloudinary","firebase","django","flask","sql","nosql",
    "docker","kubernetes","git","github","redux","tailwind","tailwindcss","typescript","ci/cd","ci cd",
    "jwt","oauth","mongoose","material ui","postman","vercel","render"
]

def clean_text(s: Optional[Any]) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r", " ").replace("\n", " ").strip()
    s = re.sub(r"(_x000D_|\\x[\da-fA-F]{2}|Ã..|â..|[\u2018\u2019\u201c\u201d])+"," ", s)
    s = re.sub(r"\s{2,}", " ", s)
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1].strip()
    return s

def to_native(x):
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return to_native(x.item())
    except Exception:
        pass
    return x

# ---------------- PDF extraction ----------------
def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    text_parts = []
    if _HAS_PDFPLUMBER:
        try:
            import io
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for p in pdf.pages:
                    page_text = p.extract_text() or ""
                    text_parts.append(page_text)
            if text_parts:
                return "\n".join(text_parts)
        except Exception as e:
            logger.warning("pdfplumber extraction failed: %s", e)
    if _HAS_PYPDF2:
        try:
            import io
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for p in reader.pages:
                try:
                    page_text = p.extract_text() or ""
                    text_parts.append(page_text)
                except Exception:
                    continue
            if text_parts:
                return "\n".join(text_parts)
        except Exception as e:
            logger.warning("PyPDF2 extraction failed: %s", e)
    try:
        return file_bytes.decode("utf-8", errors="replace")
    except Exception:
        return ""

# ---------------- data paths ----------------
HERE = Path(__file__).parent.resolve()
DEFAULT_DATA_DIR = HERE / "data"
cand1 = DEFAULT_DATA_DIR / "resources_clean.csv"
cand2 = DEFAULT_DATA_DIR / "resources.csv"
DATA_PATH = None
env_dp = os.getenv("DATA_PATH")
if env_dp:
    DATA_PATH = Path(env_dp)
else:
    if cand1.exists():
        DATA_PATH = cand1
    elif cand2.exists():
        DATA_PATH = cand2

logger.info("Resolved DATA_PATH -> %s", str(DATA_PATH) if DATA_PATH else "None (no CSV found)")

# ---------------- global stores ----------------
RESOURCES_DF: pd.DataFrame = pd.DataFrame()
RESOURCE_TEXTS: List[str] = []
RESOURCE_EMBEDDINGS = None
MODEL: Optional[SentenceTransformer] = None
CSV_SKILLS = set()
KNOWN_SKILLS: List[str] = []
KNOWN_EMBS = None

# ---------------- delimiter detection + CSV load ----------------
def detect_delimiter(sample_bytes: bytes) -> str:
    try:
        sample = sample_bytes.decode("utf-8", errors="replace")
    except Exception:
        try:
            sample = sample_bytes.decode("latin1", errors="replace")
        except Exception:
            sample = ""
    counts = {
        ",": sample.count(","),
        "\t": sample.count("\t"),
        "|": sample.count("|"),
        ";": sample.count(";")
    }
    # prefer tabs if prominent
    if counts["\t"] >= max(counts.values()) and counts["\t"] > 0:
        return "\t"
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample[:4096])
        return dialect.delimiter
    except Exception:
        delim = max(counts, key=counts.get)
        return delim if counts[delim] > 0 else ","

def load_resources(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    raw_bytes = None
    try:
        with open(path, "rb") as f:
            raw_bytes = f.read(65536)
    except Exception as e:
        logger.warning("Could not read CSV bytes for detection: %s", e)
        raw_bytes = None

    delim = ","
    if raw_bytes:
        try:
            delim = detect_delimiter(raw_bytes)
            logger.info("Auto-detected CSV delimiter: '%s' for file %s", delim, str(path))
        except Exception as e:
            logger.warning("Delimiter detection failed: %s", e)
            delim = ","

    # try several separators if needed
    read_attempts = [delim]
    if delim != "\t":
        read_attempts.append("\t")
    if "," not in read_attempts:
        read_attempts.append(",")

    df = pd.DataFrame()
    for sep_try in read_attempts:
        try:
            df = pd.read_csv(path, sep=sep_try, dtype=str, keep_default_na=False, encoding="utf-8", engine="python", on_bad_lines="skip")
            logger.info("Loaded CSV with sep='%s' -> rows: %d, cols: %d", sep_try, len(df), len(df.columns))
            if len(df.columns) == 1:
                sample_col_vals = df.iloc[:, 0].astype(str)
                if sample_col_vals.str.contains("\t").any():
                    logger.info("Single-column CSV contains tabs — splitting column by tab characters.")
                    split_df = df.iloc[:, 0].str.split("\t", expand=True)
                    split_df.columns = [f"col{i}" for i in range(split_df.shape[1])]
                    df = split_df
            df = df.dropna(axis=1, how="all")
            if not df.empty:
                break
        except Exception as e:
            logger.warning("read_csv failed with sep='%s': %s", sep_try, e)
            df = pd.DataFrame()
            continue

    if df.empty:
        logger.error("Failed to read CSV at %s with tried separators %s", path, read_attempts)
        return pd.DataFrame()

    # clean cells
    for c in df.columns:
        df[c] = df[c].apply(lambda v: clean_text(v) if not pd.isna(v) else "")

    # normalize column names and remove duplicates
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    # relaxed: keep any row with at least one non-empty cell
    try:
        df = df[df.astype(bool).any(axis=1)].reset_index(drop=True)
    except Exception:
        pass

    logger.info("After normalization, final resource rows: %d, columns: %s", len(df), df.columns.tolist())
    return df

def build_resource_texts(df: pd.DataFrame) -> List[str]:
    texts = []
    if df is None or df.empty:
        return texts
    for _, row in df.iterrows():
        pieces = []
        for col in df.columns:
            v = row.get(col, "")
            if isinstance(v, str) and v.strip():
                pieces.append(v.strip())
        text_blob = " | ".join(pieces)
        text_blob = clean_text(text_blob)
        texts.append(text_blob)
    return texts

def learn_csv_skills(df: pd.DataFrame):
    s = set()
    if df is None or df.empty:
        return s
    for col in df.columns:
        if "skill" in col.lower() or "topics" in col.lower() or "course" in col.lower():
            for val in df[col].dropna().astype(str).tolist():
                for token in re.split(r"[,;/\|]", val):
                    token = clean_text(token).lower().strip()
                    if 1 < len(token) <= 80 and token not in STOPWORDS and token not in EXTRA_BLACKLIST:
                        s.add(token)
    for col in df.columns:
        if any(x in col.lower() for x in ("title", "short", "desc", "name")):
            for val in df[col].dropna().astype(str).tolist():
                for phrase in TECH_PHRASES:
                    if phrase in val.lower():
                        s.add(phrase)
    return s

# ---------- initialize model & resources ----------
def initialize():
    global MODEL, RESOURCES_DF, RESOURCE_TEXTS, RESOURCE_EMBEDDINGS, CSV_SKILLS, KNOWN_SKILLS, KNOWN_EMBS
    if DATA_PATH:
        RESOURCES_DF = load_resources(DATA_PATH)
    else:
        RESOURCES_DF = pd.DataFrame()
    RESOURCE_TEXTS = build_resource_texts(RESOURCES_DF)
    logger.info("CSV rows text built: %s", len(RESOURCE_TEXTS))
    CSV_SKILLS = learn_csv_skills(RESOURCES_DF)

    # load SBERT
    model_name = os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    logger.info("Loading SBERT model: %s", model_name)
    try:
        MODEL = SentenceTransformer(model_name)
        logger.info("SBERT model loaded.")
    except Exception as e:
        logger.exception("Failed to load SBERT model: %s", e)
        MODEL = None

    if MODEL and RESOURCE_TEXTS:
        try:
            RESOURCE_EMBEDDINGS = MODEL.encode(RESOURCE_TEXTS, convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True)
            logger.info("Resource embeddings computed: %s", getattr(RESOURCE_EMBEDDINGS, "shape", None))
        except Exception as e:
            logger.exception("Failed to compute resource embeddings: %s", e)
            RESOURCE_EMBEDDINGS = None
    else:
        RESOURCE_EMBEDDINGS = None
        if not RESOURCE_TEXTS:
            logger.warning("No resource texts found or MODEL missing; recommendations disabled until CSV + model are available.")

    # spaCy & known skill embeddings
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model loaded.")
    except Exception as e:
        logger.exception("Failed to load spaCy model: %s", e)
        nlp = None

    KNOWN_SKILLS = sorted(list(CSV_SKILLS.union(set(TECH_PHRASES))))
    if MODEL and KNOWN_SKILLS:
        try:
            KNOWN_EMBS = MODEL.encode(KNOWN_SKILLS, convert_to_tensor=True, normalize_embeddings=True)
            logger.info("Known skills embeddings computed: %s", len(KNOWN_SKILLS))
        except Exception as e:
            logger.exception("Failed to compute KNOWN_SKILLS embeddings: %s", e)
            KNOWN_EMBS = None
    else:
        KNOWN_EMBS = None

    # init semantic extractor helper
    try:
        if nlp and MODEL:
            semx.init(nlp, MODEL, KNOWN_SKILLS, KNOWN_EMBS)
            logger.info("semantic_extractor initialized with %d known skills", len(KNOWN_SKILLS))
        else:
            logger.warning("semantic_extractor not initialized (nlp or MODEL missing).")
    except Exception as e:
        logger.exception("Failed to init semantic_extractor: %s", e)

initialize()

# ---------------- utility: extract recommendation from CSV row ----------------
def extract_recommendation_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(row, dict):
        return {"title": str(row)}
    def _get(keys):
        for k in keys:
            if k in row and row[k]:
                return clean_text(row[k])
        return ""
    title = _get(["Title", "title", "Course Title", "Course", "Name", "Course Title"]) or ""
    url = _get(["URL", "Url", "Link", "Course URL", "Course Link"])
    desc = _get(["Short Intro", "Description", "Course Short Intro", "What you learn"])
    platform = _get(["Site", "Platform", "School", "Provider"])
    rating_raw = _get(["Rating", "Stars", "Average Rating", "Number of ratings"])
    duration = _get(["Duration", "Weekly study", "Approx. time", "Monthly access", "4-Month access"])
    try:
        rating = float(re.sub(r"[^\d\.]", "", rating_raw)) if rating_raw else None
    except Exception:
        rating = None
    if not title:
        if desc:
            title = " ".join(desc.split()[:6]) + "..."
        else:
            title = ""
    return {
        "title": title,
        "url": url,
        "desc": desc,
        "platform": platform,
        "rating": rating,
        "duration": duration
    }

# ---------------- ngram extractor ----------------
def generate_ngrams(text: str, n_max: int = 3):
    tokens = [t.strip().lower() for t in re.split(r"[\W_]+", text) if t.strip()]
    out = set()
    L = len(tokens)
    for n in range(1, n_max + 1):
        for i in range(0, L - n + 1):
            gram = " ".join(tokens[i : i + n])
            out.add(gram)
    return out

def extract_candidate_skills_from_text(text: str) -> List[str]:
    text = clean_text(text)
    ngrams = generate_ngrams(text, n_max=3)
    candidates = set()
    for ng in ngrams:
        if ng in CSV_SKILLS:
            candidates.add(ng)
        elif ng in TECH_PHRASES:
            candidates.add(ng)
        else:
            parts = ng.split()
            if any(p in TECH_PHRASES for p in [" ".join(parts[i:i+2]) for i in range(len(parts)-1)]) :
                candidates.add(ng)
            else:
                if len(ng) > 2 and not any(w in STOPWORDS or w in EXTRA_BLACKLIST for w in ng.split()):
                    if len(ng) <= 40:
                        candidates.add(ng)
    final = [c for c in sorted(candidates) if not re.match(r"^\d+$", c)]
    return final

# ---------------- core analysis ----------------
def analyze_texts(job_desc_text: str, resume_text: str = "", top_k_recs: int = 8) -> Dict[str, Any]:
    t0 = time.time()
    if MODEL is None:
        logger.error("No SBERT model available.")
        return {"data": {"match": {"matched_skills": [], "missing_skills": [], "match_percent": 0.0, "context_similarity": None}, "recommendations": []}}

    jd_text = clean_text(job_desc_text or "")
    jd_sentences = [s.strip() for s in re.split(r"[.\n;]\s*", jd_text) if s.strip()]

    jd_embedding_mean = None
    if jd_sentences and MODEL:
        try:
            jd_embeddings = MODEL.encode(jd_sentences, convert_to_tensor=True)
            jd_embedding_mean = jd_embeddings.mean(dim=0)
        except Exception as e:
            logger.exception("JD embedding failed: %s", e)
            jd_embedding_mean = None

    recs = []
    match_obj = {"matched_skills": [], "missing_skills": [], "match_percent": 0.0, "context_similarity": None}

    # Recommendations via cosine sim (only if we have embeddings)
    if jd_embedding_mean is not None and RESOURCE_EMBEDDINGS is not None and getattr(RESOURCE_EMBEDDINGS, "shape", (0,))[0] > 0:
        try:
            cos_scores = util.cos_sim(jd_embedding_mean, RESOURCE_EMBEDDINGS)
            cos_arr = cos_scores.detach().cpu().numpy().flatten()
            logger.info("Sample cos_sim (first 5): %s", cos_arr[:5].tolist())
            top_k = min(10, len(cos_arr))
            if top_k > 0:
                topk_vals = np.sort(cos_arr)[-top_k:]
                topk_norm = [((float(x) + 1) / 2.0) for x in topk_vals]
                match_obj["match_percent"] = round(float(np.mean(topk_norm)) * 100, 2)
                match_obj["context_similarity"] = float(np.max(cos_arr))
            rec_indices = np.argsort(-cos_arr)[:top_k_recs]
            for idx in rec_indices:
                score = float(cos_arr[idx])
                raw_row = {}
                if not RESOURCES_DF.empty and idx < len(RESOURCES_DF):
                    raw_row = RESOURCES_DF.iloc[int(idx)].to_dict()
                else:
                    raw_row = {"Title": RESOURCE_TEXTS[idx] if idx < len(RESOURCE_TEXTS) else ""}
                rec = extract_recommendation_from_row(raw_row)
                if (not rec["title"].strip() and not rec["desc"].strip()) or (not rec.get("url") and not rec["desc"].strip()):
                    continue
                rec["score_percent"] = round(((score + 1) / 2.0) * 100, 2)
                rec = {k: to_native(v) for k, v in rec.items()}
                recs.append(rec)
        except Exception as e:
            logger.exception("Similarity computation failed: %s", e)
            recs = []
    else:
        logger.info("RESOURCE_EMBEDDINGS not available or empty; skipping recommendations.")

    # Semantic extraction (semantic_extractor helper)
    try:
        matched_list, missing_list = semx.semantic_filter(
            jd_text,
            resume_text,
            top_k=30,
            jd_res_threshold=0.60,
            canon_cutoff=0.80,
            keep_tech_only=True
        )
    except Exception as e:
        logger.exception("Semantic extraction failed: %s", e)
        matched_list, missing_list = [], []

    # Promote resume tokens
    def normalize_token(t):
        return re.sub(r"[^a-z0-9\+\#\.]+", " ", t.lower()).strip()

    resume_tokens = set(normalize_token(s) for s in extract_candidate_skills_from_text(resume_text))
    for tok in resume_tokens:
        if tok and (tok in (p.lower() for p in TECH_PHRASES) or tok in (s.lower() for s in matched_list) or tok in (s.lower() for s in missing_list)):
            if tok not in (m.lower() for m in matched_list):
                matched_list.append(tok)

    # canonicalization + blacklist
    LOCAL_CANON = {
        "express": "express.js", "express js": "express.js",
        "node": "node.js", "node js": "node.js", "mongo": "mongodb",
        "mongoose": "mongodb", "tailwind": "tailwindcss", "tailwind css": "tailwindcss",
        "tailwind css backend": "tailwindcss", "jwt/oauth": "jwt", "jwt oauth": "jwt",
        "docker basic understanding": "docker", "docker basics": "docker",
        "aws / firebase exposure": "firebase"
    }
    LOCAL_BLACKLIST = set(["technologies", "test", "ui design"]) | EXTRA_BLACKLIST

    def canonicalize_and_dedupe(lst):
        out = []
        seen = set()
        for s in lst:
            s0 = s.strip().lower()
            if not s0:
                continue
            if len(s0.split()) > 6:
                continue
            if s0 in LOCAL_BLACKLIST:
                continue
            canon = LOCAL_CANON.get(s0, s0)
            if canon in seen:
                continue
            seen.add(canon)
            out.append(canon)
        return out

    matched_list = canonicalize_and_dedupe(matched_list)
    missing_list = canonicalize_and_dedupe(missing_list)

    # ensure matched/missing disjoint
    matched_set = set(matched_list)
    final_missing = [m for m in missing_list if m not in matched_set]

    # fallback match percent if resource-based not computed
    if match_obj["match_percent"] == 0.0:
        denom = len(matched_list) + len(final_missing)
        match_obj["match_percent"] = round((len(matched_list) / denom) * 100, 2) if denom > 0 else 0.0
        if RESOURCE_EMBEDDINGS is None or getattr(RESOURCE_EMBEDDINGS, "shape", (0,))[0] == 0:
            match_obj["context_similarity"] = None

    match_obj["matched_skills"] = sorted(matched_list, key=lambda s: s.lower())
    match_obj["missing_skills"] = sorted(final_missing, key=lambda s: s.lower())

    return {
        "data": {
            "match": match_obj,
            "recommendations": {"recommended": recs}
        }
    }

# ---------------- endpoints ----------------
@app.get("/")
async def root():
    return {"status": True, "resources": len(RESOURCE_TEXTS), "model_loaded": MODEL is not None}

@app.get("/debug/resources")
async def debug_resources():
    sample_titles = []
    if not RESOURCES_DF.empty:
        for i, r in RESOURCES_DF.head(10).iterrows():
            title = r.get("Title") or r.get("title") or r.get("Course Title") or ""
            url = r.get("URL") or r.get("Url") or r.get("Link") or r.get("Course URL") or ""
            sample_titles.append({"title": title, "url": url})
    emb_shape = None
    if RESOURCE_EMBEDDINGS is not None:
        try:
            emb_shape = tuple(RESOURCE_EMBEDDINGS.shape)
        except Exception:
            emb_shape = str(type(RESOURCE_EMBEDDINGS))
    return {
        "DATA_PATH": str(DATA_PATH) if DATA_PATH else None,
        "resources_count": len(RESOURCE_TEXTS),
        "sample_titles": sample_titles,
        "first_5_resource_texts": RESOURCE_TEXTS[:5],
        "resource_embeddings_shape": emb_shape,
        "known_skills_count": len(KNOWN_SKILLS)
    }

@app.post("/api/analyze")
async def api_analyze(
    resume_pdf: Optional[UploadFile] = File(None),
    jd_text: Optional[str] = Form(None),
):
    try:
        resume_text = ""
        if resume_pdf is not None:
            file_bytes = await resume_pdf.read()
            resume_text = extract_text_from_pdf_bytes(file_bytes)
            logger.info("Extracted resume text length: %s", len(resume_text))

        jd_text_val = clean_text(jd_text or "")
        if not jd_text_val and resume_text:
            logger.info("No JD provided; using resume text as fallback for analysis.")
            jd_text_val = resume_text[:4000]

        if not jd_text_val:
            return JSONResponse(status_code=400, content={"error": "No job description text provided. Provide jd_text or upload resume."})

        raw = analyze_texts(jd_text_val, resume_text=resume_text, top_k_recs=8)
        normalized = {
            "match": raw.get("data", {}).get("match", {}),
            "recommendations": raw.get("data", {}).get("recommendations", {}).get("recommended", [])
        }
        if "context_similarity" in normalized["match"] and normalized["match"]["context_similarity"] is None:
            normalized["match"]["context_similarity"] = None
        normalized["_meta"] = {
            "model_name": os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            "resources_count": len(RESOURCE_TEXTS)
        }
        return JSONResponse(content=normalized)
    except Exception as e:
        logger.exception("Analyze endpoint error: %s", e)
        return JSONResponse(status_code=500, content={"error": "server error", "detail": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 9000)), reload=True)
