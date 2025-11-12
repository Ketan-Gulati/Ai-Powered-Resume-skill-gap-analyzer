import os, io, json, traceback
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
from utils.extract_text import extract_text_from_pdf

BASE = os.path.dirname(_file_)
DATA_PATH = os.path.join(BASE, "data", "resources_clean.csv")
MODEL_NAME = os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI(title="Skill Gap Analyzer - Backend")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173",""], allow_methods=[""], allow_headers=["*"])

# --- utility: safe CSV loader (robust to messy CSVs) ---
def load_resources(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"resources file not found at {path}")
    # Try pandas auto-detect, if fails, fallback to tab-split sanitizer or python engine
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8", header=None)
        except Exception as e:
            raise RuntimeError("Failed to parse CSV: " + str(e))
    # normalize columns: create title & description if not present
    # If headerless file (many kaggle resources have columns), attempt to guess
    if "title" not in df.columns:
        # attempt to map known positions
        if df.shape[1] >= 2:
            df.columns = [f"c{i}" for i in range(df.shape[1])]
            df["title"] = df.iloc[:,1].astype(str).fillna("")
            df["description"] = df.iloc[:,2].astype(str).fillna("") if df.shape[1] > 2 else df["title"]
        else:
            df["title"] = df.iloc[:,0].astype(str).fillna("")
            df["description"] = df["title"]
    else:
        df["title"] = df["title"].fillna("").astype(str)
        df["description"] = df.get("description","").fillna("").astype(str)
    df["text"] = (df["title"].astype(str) + " - " + df["description"].astype(str)).str.strip()
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    return df

# --- embedding backend: try SBERT, else fallback to TF-IDF ---
SBERT_AVAILABLE = True
embedder = None
try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(MODEL_NAME)
    print("SBERT loaded:", MODEL_NAME)
except Exception as e:
    print("SBERT not available, falling back to TF-IDF. Error:", e)
    SBERT_AVAILABLE = False

# Load resources
df = load_resources(DATA_PATH)
texts = df["text"].tolist()
n_resources = len(texts)
print("Loaded resources rows:", n_resources)

# Build vectors
vectorizer = None
resource_vectors = None
nbrs = None
if SBERT_AVAILABLE and embedder:
    try:
        resource_vectors = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        nbrs = NearestNeighbors(n_neighbors=min(10, max(1, n_resources)), metric="cosine").fit(resource_vectors)
        print("Built SBERT embeddings.")
    except Exception as e:
        print("SBERT embedding failed, falling back. Error:", e)
        SBERT_AVAILABLE = False

if not SBERT_AVAILABLE:
    # TF-IDF fallback with progressive attempts to avoid empty vocab
    try:
        vectorizer = TfidfVectorizer(max_features=30000, stop_words="english")
        resource_vectors = vectorizer.fit_transform(texts)
        nbrs = NearestNeighbors(n_neighbors=min(10, max(1, n_resources)), metric="cosine").fit(resource_vectors)
        print("Built TF-IDF vectors.")
    except Exception as e:
        print("TF-IDF failed:", e)
        try:
            # Last-ditch: CountVectorizer with no stop words
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer()
            resource_vectors = vectorizer.fit_transform(texts)
            nbrs = NearestNeighbors(n_neighbors=min(10, max(1, n_resources)), metric="cosine").fit(resource_vectors)
            print("Built CountVectorizer vectors.")
        except Exception as e2:
            print("No text vectors available. Search will return fallback results.", e2)
            resource_vectors = None
            nbrs = None

def compute_text_vector(text):
    text = (text or "").strip()
    if not text:
        return None
    if SBERT_AVAILABLE and embedder:
        return embedder.encode([text], convert_to_numpy=True)[0]
    if vectorizer is not None:
        return vectorizer.transform([text])
    return None

@app.post("/api/analyze")
async def analyze(jd_text: str = Form(...), resume_pdf: UploadFile = File(None)):
    try:
        resume_text = ""
        if resume_pdf is not None:
            raw = await resume_pdf.read()
            resume_text = extract_text_from_pdf(io.BytesIO(raw))
        # compute context similarity: JD vs resume
        jd_vec = compute_text_vector(jd_text)
        resume_vec = compute_text_vector(resume_text) if resume_text else None
        context_sim = 0.0
        if jd_vec is not None and resume_vec is not None:
            try:
                # ensure shapes compatible
                if hasattr(jd_vec, "reshape"):
                    cos = cosine_similarity(jd_vec.reshape(1,-1), resume_vec.reshape(1,-1))[0,0]
                else:
                    cos = float(np.dot(jd_vec, resume_vec) / (np.linalg.norm(jd_vec)*np.linalg.norm(resume_vec)+1e-9))
                context_sim = max(0.0, min(1.0, float(cos)))
            except Exception:
                context_sim = 0.0
        # skill extraction (simple, improved later): use SBERT to semantically find candidate skills from JD
        # For now, split by common separators and also keep noun-like tokens (short approach)
        tokens = [t.strip() for t in jd_text.replace("/",",").replace("|",",").replace("•",",").split(",") if t.strip()]
        # keep top candidates
        skill_candidates = []
        for t in tokens:
            t = t.strip()
            if len(t) > 1:
                skill_candidates.append(t.lower())
        skill_candidates = list(dict.fromkeys(skill_candidates))[:20]
        resume_lower = (resume_text or "").lower()
        matched, missing = [], []
        for s in skill_candidates:
            if resume_lower and s in resume_lower:
                matched.append(s)
            else:
                missing.append(s)
        # recommendations: search nearest resources semantically using missing-augmented query
        query_text = jd_text + " " + " ".join(missing[:5])
        q_vec = compute_text_vector(query_text)
        recs = []
        if q_vec is not None and nbrs is not None:
            try:
                if hasattr(q_vec, "reshape"):
                    dists, idxs = nbrs.kneighbors(q_vec.reshape(1,-1), n_neighbors=min(6, n_resources))
                else:
                    dists, idxs = nbrs.kneighbors(q_vec, n_neighbors=min(6, n_resources))
                for dist, idx in zip(dists[0], idxs[0]):
                    row = df.iloc[int(idx)].to_dict()
                    score = float((1.0 - dist) * 100)
                    row["score_percent"] = round(score,2)
                    recs.append(row)
            except Exception:
                recs = []
        # fallback if no recs
        if not recs:
            # return top rows as fallback
            for i in range(min(6, n_resources)):
                r = df.iloc[i].to_dict()
                r["score_percent"] = None
                recs.append(r)
        return {"data": {"match": {"context_similarity": round(context_sim*100,2),
                                   "matched_skills": matched, "missing_skills": missing},
                         "recommendations": recs}}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommend_resources")
async def recommend_resources(payload: dict):
    try:
        jd_text = payload.get("jd_text","")
        missing = payload.get("missing_skills",[])
        top_k = int(payload.get("top_k",6))
        query_text = jd_text + " " + " ".join(missing[:5])
        q_vec = compute_text_vector(query_text)
        recs = []
        if q_vec is not None and nbrs is not None:
            dists, idxs = nbrs.kneighbors(q_vec.reshape(1,-1), n_neighbors=min(top_k, n_resources))
            for dist, idx in zip(dists[0], idxs[0]):
                row = df.iloc[int(idx)].to_dict()
                row["score_percent"] = round(float((1.0-dist)*100),2)
                recs.append(row)
        return {"results": recs}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))