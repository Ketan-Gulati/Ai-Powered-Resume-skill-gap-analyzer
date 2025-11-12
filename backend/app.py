from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pdfplumber
import io, numpy as np, os, traceback

app = FastAPI(title="Skill Gap Analyzer - AI Backend")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset and model
BASE = os.path.dirname(_file_)
DATA_PATH = os.path.join(BASE, "data", "resources.csv")

print("Loading resources from:", DATA_PATH)
df = pd.read_csv(DATA_PATH).fillna("")
df["text"] = (df["title"] + " - " + df["description"]).str.strip()

print("Loading model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model ready ✔")

embeddings = model.encode(df["text"].tolist(), convert_to_numpy=True)

@app.post("/api/analyze")
async def analyze(jd_text: str = Form(...), resume_pdf: UploadFile = File(...)):
    try:
        resume_text = ""
        with pdfplumber.open(io.BytesIO(await resume_pdf.read())) as pdf:
            for page in pdf.pages:
                resume_text += page.extract_text() or ""

        # Encode
        jd_emb = model.encode([jd_text])[0]
        res_emb = model.encode([resume_text])[0]

        # Context similarity
        sim = float(cosine_similarity([jd_emb], [res_emb])[0][0]) * 100

        # Skill extraction (simple NLP)
        tokens = [t.lower().strip() for t in jd_text.split() if len(t) > 2]
        matched, missing = [], []
        for skill in tokens:
            if skill in resume_text.lower():
                matched.append(skill)
            else:
                missing.append(skill)

        # Recommendation (based on JD context)
        q_emb = model.encode([jd_text])[0]
        sims = cosine_similarity([q_emb], embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:5]
        recommendations = df.iloc[top_idx][["title", "url", "description"]].to_dict(orient="records")

        return {
            "context_similarity": round(sim, 2),
            "matched_skills": matched[:10],
            "missing_skills": missing[:10],
            "recommendations": recommendations
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}