# app.py
import os
import io
import json
import traceback
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np

# NLP / ML feature imports - import heavy libs lazily / safely
SBERT_AVAILABLE = True
embedder = None
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SBERT_AVAILABLE = False
    embedder = None
    print("ML libs not available or SBERT import failed:", repr(e))

# scikit-learn imports (TF-IDF / cosine)
from sklearn.metrics.pairwise import cosine_similarity

# Vectorizers (fallbacks)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# NearestNeighbors only used when SBERT embeddings are available
try:
    from sklearn.neighbors import NearestNeighbors
except Exception:
    NearestNeighbors = None

# PDF extraction
import pdfplumber

# App
app = FastAPI(title="Skill Gap Analyzer - Backend (robust)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = os.path.dirname(_file_)
# Use cleaned CSV placed in backend/data/resources_clean.csv (you said you created that)
DATA_PATH = os.path.join(BASE, "data", "resources_clean.csv")

# Model name (if SBERT available)
SBERT_MODEL = os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Globals to hold resources & vectors
df: Optional[pd.DataFrame] = None
texts: List[str] = []
resource_embeddings = None
resource_tfidf = None
vectorizer = None
nbrs = None


def safe_read_csv(path: str) -> pd.DataFrame:
    """
    Attempt robust CSV loading:
    1) try autodetect (sep=None, engine='python')
    2) try tab-separated
    3) fall back to reading lines and splitting on '\t' or commas minimally
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Resources file not found at {path}")

    # Attempt 1: autodetect
    try:
        df_local = pd.read_csv(path, sep=None, engine="python", encoding="utf-8", header=0)
        print("CSV auto-detect success. Columns:", df_local.shape)
        return df_local
    except Exception as e:
        print("CSV auto-detect failed:", e)

    # Attempt 2: tab-separated
    try:
        df_local = pd.read_csv(path, sep="\t", engine="python", encoding="utf-8", header=None)
        print("CSV loaded as tab-separated. Rows:", len(df_local))
        return df_local
    except Exception as e:
        print("CSV tab-load failed:", e)

    # Attempt 3: naive line split -> build DataFrame with fallback columns
    try:
        rows = []
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip("\n\r")
                # if it contains tabs, split by tabs else split by comma
                if "\t" in line:
                    parts = line.split("\t")
                else:
                    parts = line.split(",")
                rows.append(parts)
        # create dataframe, pad rows to equal length
        maxc = max(len(r) for r in rows) if rows else 0
        rows2 = [r + [""] * (maxc - len(r)) for r in rows]
        df_local = pd.DataFrame(rows2)
        print("CSV fallback read. Rows:", len(df_local), "Cols:", df_local.shape[1])
        return df_local
    except Exception as e:
        print("CSV fallback also failed:", e)
        raise


def load_resources(path: str):
    global df, texts, resource_embeddings, resource_tfidf, vectorizer, nbrs

    print("DATA_PATH ->", path)
    df_local = safe_read_csv(path)

    # Normalize columns: try to find title/description columns if present, else combine all text columns
    # Heuristic: common column names
    colnames = [c.lower() for c in df_local.columns.astype(str)]
    title_col = None
    desc_col = None
    for c in df_local.columns:
        lc = str(c).lower()
        if "title" in lc and title_col is None:
            title_col = c
        if "description" in lc and desc_col is None:
            desc_col = c
        if "url" in lc and "link" not in lc:
            # ignore url for text
            pass

    # If no title/description columns, combine textual columns (object dtype)
    def make_text_row(row):
        pieces = []
        if title_col is not None:
            pieces.append(str(row.get(title_col, "") or ""))
        if desc_col is not None:
            pieces.append(str(row.get(desc_col, "") or ""))
        # add any other string-like columns that look useful (limit to first 3)
        if title_col is None and desc_col is None:
            # fallback: join all string columns
            for k, v in row.items():
                if isinstance(v, str) and v.strip():
                    pieces.append(v.strip())
        return " - ".join([p.strip() for p in pieces if p is not None and str(p).strip()])

    # Ensure df_local is a dataframe
    df_local = df_local.fillna("")
    df_local["text"] = df_local.apply(make_text_row, axis=1)
    texts_local = df_local["text"].astype(str).tolist()
    # strip empty
    texts_local = [t.strip() for t in texts_local]

    # assign globals
    df = df_local
    texts = texts_local
    print("Loaded resources rows:", len(texts))

    # Try to load SBERT embedder if available
    if SBERT_AVAILABLE:
        try:
            global embedder
            if embedder is None:
                print("Loading SBERT model:", SBERT_MODEL)
                embedder = SentenceTransformer(SBERT_MODEL)
                print("SBERT model loaded.")
            # compute resource embeddings (silent if no texts)
            if texts:
                resource_embeddings = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
                # fit NearestNeighbors if available
                if NearestNeighbors is not None:
                    nbrs = NearestNeighbors(n_neighbors=min(10, len(texts)), metric="cosine").fit(resource_embeddings)
                print("Resource embeddings computed:", resource_embeddings.shape)
                return
        except Exception as e:
            print("SBERT model failed to load/encode:", repr(e))
            # fallback to vectorizers below

    # If SBERT not available or failed, fallback to TF-IDF or CountVectorizer
    print("Falling back to TF-IDF / CountVectorizer.")
    # Try TF-IDF (with light preprocessing)
    try:
        # Minimal custom tokenizer/preproc is left to sklearn defaults
        vectorizer_local = TfidfVectorizer(max_features=20000, stop_words="english")
        if any(t.strip() for t in texts):
            resource_tfidf_local = vectorizer_local.fit_transform(texts)
            if resource_tfidf_local.shape[1] == 0:
                raise ValueError("empty vocabulary after TF-IDF fit")
            vectorizer = vectorizer_local
            resource_tfidf = resource_tfidf_local
            print("TF-IDF matrix shape:", resource_tfidf.shape)
            return
        else:
            raise ValueError("No resource texts available for TF-IDF")
    except Exception as e:
        print("TF-IDF failed:", repr(e))

    # Try CountVectorizer
    try:
        vectorizer_local = CountVectorizer(max_features=20000)
        resource_tfidf_local = vectorizer_local.fit_transform(texts)
        vectorizer = vectorizer_local
        resource_tfidf = resource_tfidf_local
        print("CountVectorizer matrix shape:", resource_tfidf.shape)
        return
    except Exception as e:
        print("CountVectorizer failed:", repr(e))

    # If all fail, keep them None and the search will return fallback items
    print("No text vectors available. Search will return fallback results.")


# initialize on startup
try:
    load_resources(DATA_PATH)
except Exception as e:
    print("Initial load_resources failed:", repr(e))


def compute_text_embedding(text: str):
    """
    Returns numpy vector for the input text using embedder (SBERT) or vectorizer (TF-IDF).
    If neither available, returns None.
    """
    global embedder, vectorizer
    if SBERT_AVAILABLE and embedder is not None:
        try:
            vec = embedder.encode([text], convert_to_numpy=True)[0]
            return vec
        except Exception as e:
            print("SBERT compute_text_embedding failed:", e)
            return None
    # fallback to vectorizer
    if vectorizer is not None:
        try:
            X = vectorizer.transform([text])
            # convert to dense numpy array (row vector)
            return X.toarray()[0]
        except Exception as e:
            print("Vectorizer transform failed:", e)
            return None
    return None


@app.post("/api/analyze")
async def analyze(jd_text: str = Form(...), resume_pdf: UploadFile = File(None)):
    """
    Accepts form-data: jd_text (string), optional resume_pdf upload.
    Returns match scores, matched/missing skills (simple heuristic), and recommendations.
    """
    try:
        # 1) extract resume text from PDF if provided
        resume_text = ""
        if resume_pdf is not None:
            data = await resume_pdf.read()
            try:
                with pdfplumber.open(io.BytesIO(data)) as pdf:
                    pages = [p.extract_text() or "" for p in pdf.pages]
                    resume_text = "\n".join(pages)
            except Exception as e:
                print("PDF parse error:", e)
                resume_text = ""

        # 2) compute "context similarity" between JD and resume
        context_sim_pct = 0.0
        if resume_text.strip():
            # Try SBERT -> cosine on embeddings
            if SBERT_AVAILABLE and embedder is not None:
                try:
                    emb = embedder.encode([jd_text, resume_text], convert_to_numpy=True)
                    a, b = emb[0], emb[1]
                    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
                    context_sim_pct = round(max(0.0, min(1.0, cos)) * 100.0, 2)
                except Exception as e:
                    print("SBERT cosine failed:", e)
                    context_sim_pct = 0.0
            else:
                # fallback to TF-IDF similarity if available
                q_vec = compute_text_embedding(jd_text)
                r_vec = compute_text_embedding(resume_text)
                if q_vec is not None and r_vec is not None:
                    try:
                        cos = float(
                            np.dot(q_vec, r_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(r_vec) + 1e-9)
                        )
                        context_sim_pct = round(max(0.0, min(1.0, cos)) * 100.0, 2)
                    except Exception as e:
                        print("Fallback cosine error:", e)
                        context_sim_pct = 0.0
                else:
                    context_sim_pct = 0.0
        else:
            context_sim_pct = 0.0

        # 3) simple "skill extraction" heuristic:
        #   - split JD by commas and newlines and common separators, take short tokens as skill candidates
        tokens = []
        raw = jd_text.replace("/", ",").replace("|", ",").replace("•", ",").replace("·", ",")
        parts = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
        # choose candidate phrases (trim to 2-5 words)
        skill_candidates = []
        for p in parts:
            # ignore very long sentences
            words = p.split()
            if 1 < len(words) <= 5:
                skill_candidates.append(p)
            elif len(words) == 1:
                skill_candidates.append(p)
            else:
                # split long sentence into smaller by 'and' or 'or'
                sub = []
                if " and " in p or " or " in p:
                    for s in p.replace(" and ", ",").replace(" or ", ",").split(","):
                        s = s.strip()
                        if s and len(s.split()) <= 4:
                            sub.append(s)
                if sub:
                    skill_candidates.extend(sub)
        # dedupe while preserving order
        seen = set()
        skill_candidates_clean = []
        for s in skill_candidates:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                skill_candidates_clean.append(s)
        skill_candidates = skill_candidates_clean[:30]

        # 4) matched / missing by simple substring match in resume text (lowercase)
        resume_lower = resume_text.lower()
        matched = []
        missing = []
        for s in skill_candidates:
            if resume_lower and s.lower() in resume_lower:
                matched.append(s)
            else:
                missing.append(s)

        # 5) Build query for recommendations: jd + missing skills text
        query_text = jd_text + " " + " ".join(missing)

        # 6) find recommended resources
        recommended = []
        TOP_K = 6

        # If we have SBERT embeddings and nbrs:
        if SBERT_AVAILABLE and resource_embeddings is not None and nbrs is not None:
            q_emb = compute_text_embedding(query_text)
            if q_emb is not None:
                try:
                    # kneighbors expects same dimension
                    dists, idxs = nbrs.kneighbors([q_emb], n_neighbors=min(TOP_K, resource_embeddings.shape[0]))
                    for dist, idx in zip(dists[0], idxs[0]):
                        row = df.iloc[int(idx)].to_dict()
                        score = float((1.0 - dist) * 100.0)
                        row["score_percent"] = round(score, 2)
                        recommended.append(row)
                except Exception as e:
                    print("NearestNeighbors/kneighbors failed:", e)

        # Else if we have TF-IDF matrix:
        elif resource_tfidf is not None and vectorizer is not None:
            q_vec = None
            try:
                q_vec = vectorizer.transform([query_text])
                sims = cosine_similarity(q_vec, resource_tfidf).reshape(-1)
                # pick top indices
                top_idx = np.argsort(-sims)[:TOP_K]
                for idx in top_idx:
                    if idx < len(df):
                        row = df.iloc[int(idx)].to_dict()
                        row["score_percent"] = round(float(sims[idx]) * 100.0, 2)
                        recommended.append(row)
            except Exception as e:
                print("TF-IDF recommendation failed:", e)

        # Final fallback: if nothing worked, return first N rows as fallback
        if not recommended and df is not None and len(df) > 0:
            for i in range(min(TOP_K, len(df))):
                row = df.iloc[int(i)].to_dict()
                row["score_percent"] = 0.0
                recommended.append(row)

        # prepare response
        match_percent = round((len(matched) / (len(skill_candidates) + 1)) * 100.0, 2) if skill_candidates else 0.0

        response = {
            "data": {
                "match": {
                    "context_similarity": context_sim_pct,
                    "match_percent": match_percent,
                    "matched_skills": matched,
                    "missing_skills": missing,
                },
                "recommendations": {"recommended": recommended},
            }
        }
        return response

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recommend_resources")
async def recommend_resources(payload: dict):
    """
    Accepts payload: { "jd_text": "...", "missing_skills": [...], "top_k": 6 }
    Returns list of recommended resources (same logic as above)
    """
    try:
        jd_text = payload.get("jd_text", "")
        missing = payload.get("missing_skills", []) or []
        top_k = int(payload.get("top_k", 6))

        query_text = jd_text + " " + " ".join(missing)

        recs = []
        # SBERT path
        if SBERT_AVAILABLE and resource_embeddings is not None and nbrs is not None:
            q_emb = compute_text_embedding(query_text)
            if q_emb is not None:
                try:
                    dists, idxs = nbrs.kneighbors([q_emb], n_neighbors=min(top_k, resource_embeddings.shape[0]))
                    for dist, idx in zip(dists[0], idxs[0]):
                        row = df.iloc[int(idx)].to_dict()
                        score = float((1.0 - dist) * 100.0)
                        row["score_percent"] = round(score, 2)
                        recs.append(row)
                except Exception as e:
                    print("NearestNeighbors/kneighbors failed in recommend_resources:", e)

        # TF-IDF path
        elif resource_tfidf is not None and vectorizer is not None:
            try:
                q_vec = vectorizer.transform([query_text])
                sims = cosine_similarity(q_vec, resource_tfidf).reshape(-1)
                top_idx = np.argsort(-sims)[:top_k]
                for idx in top_idx:
                    if idx < len(df):
                        row = df.iloc[int(idx)].to_dict()
                        row["score_percent"] = round(float(sims[idx]) * 100.0, 2)
                        recs.append(row)
            except Exception as e:
                print("TF-IDF recommend_resources failed:", e)

        # fallback
        if not recs and df is not None:
            for i in range(min(top_k, len(df))):
                row = df.iloc[int(i)].to_dict()
                row["score_percent"] = 0.0
                recs.append(row)

        return {"results": recs}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))