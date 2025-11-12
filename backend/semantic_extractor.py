# semantic_extractor.py

import re
import numpy as np
from sentence_transformers import util

# Globals (initialized by app)
nlp = None
MODEL = None
KNOWN_SKILLS = None        # list[str]
KNOWN_EMBS = None         # torch tensor (M, D) normalized

# ---------- conservative filtering & canonicalization ----------

TECH_TOKENS = {
    "js", "javascript", "node", "nodejs", "node.js", "react", "reactjs", "react.js",
    "next", "nextjs", "next.js", "express", "expressjs", "mongodb", "mongo", "sql",
    "mysql", "postgres", "postgresql", "redis", "docker", "kubernetes", "k8s",
    "aws", "azure", "gcp", "firebase", "tailwind", "tailwindcss", "css", "html",
    "python", "java", "c++", "c#", "typescript", "ts", "graphql", "rest", "restapi",
    "jwt", "oauth", "postman", "git", "github", "ci", "ci/cd", "pipeline", "mongoose",
    "cloudinary", "vercel", "render", "docker-compose", "json"
}

CANONICAL_MANUAL = {
    "node": "node.js",
    "node js": "node.js",
    "node.js": "node.js",
    "mongo db": "mongodb",
    "mongo": "mongodb",
    "react js": "react.js",
    "reactjs": "react.js",
    "next js": "next.js",
    "nextjs": "next.js",
    "tailwind css": "tailwindcss",
    "ci cd": "ci/cd",
    "ci/cd": "ci/cd",
    "ts": "typescript",
    "express js": "express.js",
    "expressjs": "express.js",
    "html css javascript": "javascript",
    "html css": "html/css",
    "database (dbms)": "database",
    "jwt/oauth": "jwt",
    "jwt oauth": "jwt",
    "mongoose version control": "mongodb",
    "design mongodb schemas": "mongodb",
    "github deployment": "github"
}

BLACKLIST = {
    "full stack", "full stack web developer", "experience", "responsible",
    "responsibilities", "preferred", "bonus", "passionate", "curiosity",
    "deadlines", "time management", "1–3 years", "1-3 years", "remote",
    "delhi", "india", "location", "equivalent project experience", "deploy applications",
    "design", "reusable and efficient code", "coding & programming", "coding", "developer",
    "technologies", "test", "ui design"
}

_RE_DISCARD = re.compile(
    r"(^\s*$)|"                 # empty
    r"(^\d+[-–]?\d*\s*$)|"      # numeric ranges like 1-3
    r"\b(years?|months?|days?)\b|"
    r"\b(remote|delhi|india|location|work from|on-site|onsite)\b|"
    r"\b(experience|responsible|required|preferred|bonus|passionate|team|curiosity|deadline|deadlines|equivalent)\b",
    flags=re.I
)

def init(nlp_obj, model_obj, known_skills_list, known_embs_tensor):
    """
    Initialize extractor at startup.
    known_embs_tensor should be a torch tensor (M, D), normalized.
    """
    global nlp, MODEL, KNOWN_SKILLS, KNOWN_EMBS
    nlp = nlp_obj
    MODEL = model_obj
    KNOWN_SKILLS = list(known_skills_list) if known_skills_list is not None else []
    KNOWN_EMBS = known_embs_tensor  # may be None or empty tensor
    return True

def _contains_tech_token(s: str):
    if not s:
        return False
    s2 = s.lower()
    for t in TECH_TOKENS:
        if t in s2:
            return True
    return False

def get_phrases(text: str):
    """Extract noun chunks and verb-object pairs using spaCy."""
    if not text or nlp is None:
        return []
    doc = nlp(text)
    phrases = set()
    for chunk in doc.noun_chunks:
        token = chunk.text.strip().lower()
        if 2 <= len(token) <= 60 and not re.match(r"^\d+$", token):
            phrases.add(token)
    for tok in doc:
        if tok.dep_ == "dobj" and tok.head.pos_ == "VERB":
            phrases.add(f"{tok.head.lemma_} {tok.text}".lower())
    return list(phrases)

def embed_list(texts):
    """Return embeddings as a torch tensor (normalized) using MODEL.encode(..., convert_to_tensor=True)."""
    if not texts:
        return None
    embs = MODEL.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    return embs

def _canonicalize_list(phrases, similarity_cutoff=0.80, keep_tech_only=True):
    """
    Map phrases to known skills using semantic_search; conservative.
    KNOWN_EMBS should be a torch tensor (M, D) normalized.
    """
    if not phrases:
        return []
    mapped = []
    known_lower = [k.lower() for k in KNOWN_SKILLS] if KNOWN_SKILLS else []

    remaining = []
    # manual/exact mapping and blacklist filtering
    for p in phrases:
        p_clean = p.strip().lower()
        if not p_clean or p_clean in BLACKLIST or _RE_DISCARD.search(p_clean):
            continue
        if p_clean in CANONICAL_MANUAL:
            mapped.append(CANONICAL_MANUAL[p_clean])
        elif p_clean in known_lower:
            mapped.append(KNOWN_SKILLS[known_lower.index(p_clean)])
        else:
            remaining.append(p_clean)

    # semantic mapping for remaining using util.semantic_search
    if remaining:
        if KNOWN_EMBS is not None and hasattr(KNOWN_EMBS, "shape") and KNOWN_EMBS.shape[0] > 0:
            rem_emb = embed_list(remaining)  # torch tensor
            hits = util.semantic_search(rem_emb, KNOWN_EMBS, top_k=1)
            for i, h in enumerate(hits):
                if not h:
                    if (not keep_tech_only or _contains_tech_token(remaining[i])) and remaining[i] not in BLACKLIST:
                        mapped.append(remaining[i])
                    continue
                best = h[0]
                score = float(best["score"])
                idx = int(best["corpus_id"])
                if score >= similarity_cutoff:
                    mapped.append(KNOWN_SKILLS[idx])
                else:
                    if (not keep_tech_only or _contains_tech_token(remaining[i])) and remaining[i] not in BLACKLIST:
                        mapped.append(remaining[i])
        else:
            for p in remaining:
                if (not keep_tech_only or _contains_tech_token(p)) and p not in BLACKLIST:
                    mapped.append(p)

    # dedupe, preserve canonical casing where possible
    out = []
    seen = set()
    known_lower = [k.lower() for k in KNOWN_SKILLS] if KNOWN_SKILLS else []
    for item in mapped:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        if key in known_lower:
            out.append(KNOWN_SKILLS[known_lower.index(key)])
        else:
            out.append(item)
    return out

def semantic_filter(jd_text, resume_text, top_k=15,
                    jd_res_threshold=0.55, canon_cutoff=0.80,
                    keep_tech_only=True):
    """
    Pipeline:
      - extract phrases from JD & resume
      - mark JD phrases that semantically match resume phrases (>= jd_res_threshold)
      - canonicalize matched & missing phrases conservatively
      - return (matched_final, missing_final)
    """
    jd_phrases = get_phrases(jd_text)
    resume_phrases = get_phrases(resume_text)

    if not jd_phrases:
        return [], []

    jd_embs = embed_list(jd_phrases)
    res_embs = embed_list(resume_phrases) if resume_phrases else None

    matched = []
    if res_embs is not None and jd_embs is not None:
        cos_jd_res = util.cos_sim(jd_embs, res_embs).cpu().numpy()  # (P, R)
        jd_max = cos_jd_res.max(axis=1)
        for i, score in enumerate(jd_max):
            if score >= jd_res_threshold:
                matched.append(jd_phrases[i])

    matched_canon = _canonicalize_list(matched, similarity_cutoff=canon_cutoff, keep_tech_only=keep_tech_only)
    missing_candidates = [p for p in jd_phrases if p not in matched]
    missing_canon = _canonicalize_list(missing_candidates, similarity_cutoff=(canon_cutoff - 0.03), keep_tech_only=keep_tech_only)

    matched_final = sorted(dict.fromkeys(matched_canon), key=lambda s: s.lower())[:top_k]
    missing_final = sorted(dict.fromkeys(missing_canon), key=lambda s: s.lower())[:top_k]
    return matched_final, missing_final
