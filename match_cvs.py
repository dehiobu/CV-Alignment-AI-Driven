#!/usr/bin/env python3
import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import pandas as pd

_TOKEN_PATTERN = re.compile(r"\b[\w-]+\b")


def _tokenise(doc: str) -> List[str]:
    if not doc:
        return []
    return _TOKEN_PATTERN.findall(doc)
try:  # Prefer scikit-learn when available for richer n-gram support.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover - exercised when sklearn missing.
    TfidfVectorizer = None
    cosine_similarity = None

# Readers
def read_txt(p: Path) -> str:
    return p.read_text(errors="ignore")

def read_docx(p: Path) -> str:
    from docx import Document
    doc = Document(str(p))
    return "\n".join([para.text for para in doc.paragraphs])

def read_pdf(p: Path) -> str:
    from pdfminer.high_level import extract_text
    return extract_text(str(p)) or ""

EXT_READERS = {
    ".txt": read_txt,
    ".docx": read_docx,
    ".pdf": read_pdf,
}

def load_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in EXT_READERS:
        raise ValueError(f"Unsupported file type: {ext}")
    return EXT_READERS[ext](path)


def limit_text_to_pages(
    text: str, max_pages: int = 2, chars_per_page: int = 3500
) -> Tuple[str, bool]:
    """Return text truncated to the estimated size of `max_pages`."""
    limit = max_pages * chars_per_page
    if limit <= 0 or len(text) <= limit:
        return text, False

    truncated = text[:limit]
    last_space = truncated.rfind(" ")
    if last_space > limit * 0.8:
        truncated = truncated[:last_space]
    return truncated, True

# Cleaning
def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text

# Skills
def load_skills(yaml_path: Path) -> List[str]:
    import yaml
    data = yaml.safe_load(yaml_path.read_text())
    skills: List[str] = []
    for _, arr in data.items():
        skills.extend(arr or [])
    skills = sorted(set([s.strip().lower() for s in skills if s and s.strip()]))
    return skills

def skill_coverage(text: str, skills: List[str]) -> Tuple[List[str], List[str]]:
    found: List[str] = []
    for s in skills:
        if s in text:
            found.append(s)
    missing = [s for s in skills if s not in found]
    return sorted(found), sorted(missing)

def tfidf_similarity(jd_text: str, cv_texts: List[str]) -> List[float]:
    if TfidfVectorizer is not None and cosine_similarity is not None:
        vect = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9)
        corpus = [jd_text] + cv_texts
        X = vect.fit_transform(corpus)
        sims = cosine_similarity(X[0:1], X[1:]).flatten()
        return [float(s * 100.0) for s in sims]

    # Lightweight TF-IDF fallback to avoid a hard dependency on scikit-learn.
    corpus = [jd_text] + cv_texts

    tokenised = [_tokenise(doc) for doc in corpus]
    if not any(tokenised):
        return [0.0 for _ in cv_texts]

    doc_freq: Counter[str] = Counter()
    for terms in tokenised:
        doc_freq.update(set(terms))

    doc_count = len(tokenised)
    idf = {term: math.log((1 + doc_count) / (1 + freq)) + 1.0 for term, freq in doc_freq.items()}

    def vectorise(terms: List[str]) -> dict[str, float]:
        if not terms:
            return {}
        tf = Counter(terms)
        denom = float(len(terms))
        return {term: (count / denom) * idf[term] for term, count in tf.items()}

    def cosine(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        dot = sum(weight * vec_b.get(term, 0.0) for term, weight in vec_a.items())
        norm_a = math.sqrt(sum(weight * weight for weight in vec_a.values()))
        norm_b = math.sqrt(sum(weight * weight for weight in vec_b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    vectors = [vectorise(terms) for terms in tokenised]
    base = vectors[0]
    scores = [cosine(base, vec) for vec in vectors[1:]]
    return [float(score * 100.0) for score in scores]


def extract_keywords_basic(text: str, top_n: int = 25) -> List[str]:
    """Fallback keyword extractor used when scikit-learn is unavailable."""
    tokens = [tok for tok in _tokenise(text) if len(tok) >= 3]
    if not tokens:
        return []
    counts = Counter(tokens)
    return [term for term, _ in counts.most_common(top_n)]

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Rank CVs against a JD (offline TF-IDF + skill coverage).")
    ap.add_argument("--jd", required=True, help="Path to JD file (pdf/docx/txt)")
    ap.add_argument("--cvs", required=True, help="Folder with CV files (pdf/docx/txt)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--skills", default="skills.yaml", help="YAML of skills (default: skills.yaml in cwd)")
    ap.add_argument("--topn", type=int, default=15, help="Top-N skills to display (for console)")
    ap.add_argument("--minlen", type=int, default=150, help="Minimum characters to accept a CV")
    args = ap.parse_args()

    jd_path = Path(args.jd)
    cv_dir = Path(args.cvs)
    out_path = Path(args.out)
    skills_path = Path(args.skills)

    assert jd_path.exists(), f"JD not found: {jd_path}"
    assert cv_dir.exists() and cv_dir.is_dir(), f"CV folder not found: {cv_dir}"
    assert skills_path.exists(), f"Skills YAML not found: {skills_path}"

    jd_text_raw = load_text(jd_path)
    jd_text_raw, jd_trimmed = limit_text_to_pages(jd_text_raw)
    if jd_trimmed:
        print("Truncated JD content to approximately two pages for processing.")
    if not jd_text_raw.strip():
        raise SystemExit("JD file appears to be empty after extraction.")
    jd_text = clean(jd_text_raw)

    skills = load_skills(skills_path)

    cv_files = [p for p in cv_dir.iterdir() if p.suffix.lower() in EXT_READERS]
    if not cv_files:
        raise SystemExit("No supported CV files found in folder.")
    texts_raw, kept_files = [], []
    for p in cv_files:
        try:
            t = load_text(p)
        except Exception as e:
            print(f"[WARN] Failed to read {p.name}: {e}")
            continue
        if len(t) < args.minlen:
            print(f"[WARN] Skipping {p.name}: too short ({len(t)} chars)")
            continue
        texts_raw.append(t)
        kept_files.append(p)

    if not texts_raw:
        raise SystemExit("No CVs passed the min length filter. Lower --minlen if needed.")

    texts = [clean(t) for t in texts_raw]
    sims = tfidf_similarity(jd_text, texts)

    rows = []
    for p, text, sim in zip(kept_files, texts, sims):
        matched, missing = skill_coverage(text, skills)
        cov = 100.0 * len(matched) / max(1, len(skills))
        rows.append({
            "cv_file": p.name,
            "similarity": round(sim, 2),
            "skill_coverage_pct": round(cov, 1),
            "matched_skills": ", ".join(matched),
            "missing_skills": ", ".join(missing),
            "cv_chars": len(text),
        })

    df = pd.DataFrame(rows).sort_values(["similarity", "skill_coverage_pct"], ascending=[False, False])
    df.insert(1, "similarity_pct", df["similarity"].round(2))
    df["skill_coverage_pct"] = df["skill_coverage_pct"].round(1)
    df.to_csv(out_path, index=False)
    print("\n=== Ranking ===")
    print(df[["cv_file", "similarity_pct", "skill_coverage_pct"]].head(20).to_string(index=False))

    # Gap summary
    top3 = df.head(3)
    all_missing = []
    for ms in top3["missing_skills"].tolist():
        all_missing.extend([m.strip() for m in ms.split(",") if m.strip()])
    from collections import Counter
    cnt = Counter(all_missing)
    print("\n=== Common Missing Skills (Top 10 across best CVs) ===")
    for skill, freq in cnt.most_common(10):
        print(f"{skill}  x{freq}")

if __name__ == "__main__":
    main()
