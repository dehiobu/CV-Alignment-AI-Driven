"""
Automated pipeline for watching the JD folder, refreshing skills, ranking CVs,
and collaborating with ChatGPT to propose tailored resumes.

Requirements:
  - Dependencies from requirements.txt (watchdog, openai, scikit-learn, etc.)
  - Environment variable OPENAI_API_KEY pointing to a valid OpenAI key.
  - CVs stored in C:\\Users\\denni\\Downloads (default path can be overridden).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import List

import pandas as pd
import yaml
from openai import OpenAI
try:  # Degrade gracefully when scikit-learn is not present.
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:  # pragma: no cover - optional dependency.
    TfidfVectorizer = None
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from match_cvs import (
    EXT_READERS,
    clean,
    load_skills,
    load_text,
    skill_coverage,
    tfidf_similarity,
    extract_keywords_basic,
)

try:
    import tkinter as tk
    from tkinter import messagebox, scrolledtext
except ImportError as exc:  # pragma: no cover - Tkinter ships with CPython
    raise SystemExit(
        "Tkinter is required for the confirmation window. "
        "Ensure you installed the standard Python distribution."
    ) from exc


LOG = logging.getLogger("cv_automation")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


@dataclass
class PipelineConfig:
    jd_dir: Path = Path.cwd() / "JD"
    cv_dir: Path = Path.home() / "Downloads"
    skills_path: Path = Path.cwd() / "skills.yaml"
    report_path: Path = Path.cwd() / "report.csv"
    min_length: int = 500
    top_keywords: int = 25
    chat_model: str = "gpt-4o-mini"


def ensure_directories(config: PipelineConfig) -> None:
    """Guarantee that the JD directory exists so files can be dropped in."""
    if not config.jd_dir.exists():
        LOG.info("Creating JD directory at %s", config.jd_dir)
        config.jd_dir.mkdir(parents=True, exist_ok=True)


def extract_keywords(jd_text: str, top_n: int) -> List[str]:
    """Simple TF-IDF based keyword extraction."""
    if TfidfVectorizer is None:
        keywords = extract_keywords_basic(jd_text, top_n * 2)
        return keywords[:top_n]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=1,
    )
    matrix = vectorizer.fit_transform([jd_text])
    scores = matrix.toarray()[0]
    feature_names = vectorizer.get_feature_names_out()
    ranked = scores.argsort()[::-1]
    keywords: List[str] = []
    for idx in ranked:
        term = feature_names[idx].strip()
        if len(term) < 3:
            continue
        keywords.append(term)
        if len(keywords) >= top_n:
            break
    return keywords


def update_skills_yaml(skills_path: Path, new_keywords: List[str]) -> None:
    """Merge auto-extracted keywords into the skills YAML under auto_extracted."""
    current = {}
    if skills_path.exists():
        current = yaml.safe_load(skills_path.read_text()) or {}
    auto_section = sorted(set(new_keywords))
    current["auto_extracted"] = auto_section
    skills_path.write_text(yaml.safe_dump(current, sort_keys=False))
    LOG.info("Updated skills.yaml with %d auto-extracted keywords", len(auto_section))


def rank_cvs(
    jd_path: Path,
    cv_dir: Path,
    out_path: Path,
    skills_path: Path,
    min_length: int,
) -> pd.DataFrame:
    """Reuse the matching logic to score CVs."""
    if not jd_path.exists():
        raise FileNotFoundError(f"JD not found: {jd_path}")
    if not cv_dir.exists() or not cv_dir.is_dir():
        raise FileNotFoundError(f"CV folder not found: {cv_dir}")
    if not skills_path.exists():
        raise FileNotFoundError(f"Skills YAML not found: {skills_path}")

    jd_text_raw = load_text(jd_path)
    if not jd_text_raw.strip():
        raise ValueError("JD file appears empty after extraction.")
    jd_text = clean(jd_text_raw)

    skills = load_skills(skills_path)

    cv_files = [p for p in cv_dir.iterdir() if p.suffix.lower() in EXT_READERS]
    if not cv_files:
        raise ValueError("No supported CV files found in the CV directory.")

    kept_files: List[Path] = []
    texts: List[str] = []
    for cv_path in cv_files:
        try:
            raw_text = load_text(cv_path)
        except Exception as exc:  # pragma: no cover - I/O failure path
            LOG.warning("Failed to read %s: %s", cv_path.name, exc)
            continue
        if len(raw_text) < min_length:
            LOG.info("Skipping %s: too short (%d chars)", cv_path.name, len(raw_text))
            continue
        kept_files.append(cv_path)
        texts.append(clean(raw_text))

    if not texts:
        raise ValueError(
            "No CVs passed the minimum length filter. "
            "Lower the threshold or add more CVs."
        )

    similarities = tfidf_similarity(jd_text, texts)

    rows = []
    for cv_path, text, sim in zip(kept_files, texts, similarities):
        matched, missing = skill_coverage(text, skills)
        coverage = 100.0 * len(matched) / max(1, len(skills))
        rows.append(
            {
                "cv_file": cv_path.name,
                "similarity": round(sim, 2),
                "skill_coverage_pct": round(coverage, 1),
                "matched_skills": ", ".join(matched),
                "missing_skills": ", ".join(missing),
                "cv_chars": len(text),
            }
        )

    df = pd.DataFrame(rows).sort_values(
        ["similarity", "skill_coverage_pct"], ascending=[False, False]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    LOG.info("Report written to %s", out_path)
    return df


def build_initial_prompt(
    jd_text: str,
    keywords: List[str],
    ranking: pd.DataFrame,
) -> str:
    """Format a concise summary of results for ChatGPT."""
    top_rows = ranking.head(5)
    table_lines = [
        f"{row.cv_file} — similarity {row.similarity} | coverage {row.skill_coverage_pct}"
        for row in top_rows.itertuples(index=False)
    ]
    summary = "\n".join(table_lines) or "No CVs available."
    keyword_line = ", ".join(keywords[:15])
    prompt = (
        "You are assisting with CV review for a new job description.\n"
        f"Top extracted JD keywords: {keyword_line}.\n\n"
        "Here are the leading CV candidates:\n"
        f"{summary}\n\n"
        "Provide an initial comparison highlighting the strongest alignment, gaps, "
        "and whether tailoring is needed. Conclude with a clarifying question for the human reviewer, "
        "for example asking which CV should be used for an AI adoption role."
    )
    return prompt


def call_chatgpt(messages: List[dict], model: str) -> str:
    """Send conversation to ChatGPT via the OpenAI Responses API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it before running the automation."
        )
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    # Collate text output segments.
    parts = []
    for item in response.output:
        if item.type == "output_text":
            parts.append(item.text)
    return "".join(parts).strip()


def show_chat_window(conversation: List[dict], model: str) -> None:
    """Display ChatGPT output in a Tkinter window with quick yes/no replies."""
    root = tk.Tk()
    root.title("CV Matcher Assistant")
    root.geometry("720x520")

    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Segoe UI", 11))
    text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    text_area.insert(tk.END, f"Assistant:\n{conversation[-1]['content']}\n\n")
    text_area.configure(state=tk.DISABLED)

    entry_frame = tk.Frame(root)
    entry_frame.pack(fill=tk.X, padx=10, pady=5)

    entry_var = tk.StringVar()
    entry = tk.Entry(entry_frame, textvariable=entry_var, font=("Segoe UI", 11))
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def append_message(prefix: str, message: str) -> None:
        text_area.configure(state=tk.NORMAL)
        text_area.insert(tk.END, f"{prefix}:\n{message}\n\n")
        text_area.see(tk.END)
        text_area.configure(state=tk.DISABLED)

    def send_message(user_message: str) -> None:
        if not user_message.strip():
            return
        append_message("You", user_message)
        conversation.append({"role": "user", "content": user_message})
        try:
            reply = call_chatgpt(conversation, model=model)
        except Exception as exc:  # pragma: no cover - interactive path
            messagebox.showerror("ChatGPT error", str(exc))
            return
        conversation.append({"role": "assistant", "content": reply})
        append_message("Assistant", reply)

    def on_send() -> None:
        message = entry_var.get()
        entry_var.set("")
        send_message(message)

    send_btn = tk.Button(entry_frame, text="Send", command=on_send)
    send_btn.pack(side=tk.RIGHT, padx=5)

    buttons_frame = tk.Frame(root)
    buttons_frame.pack(fill=tk.X, padx=10, pady=5)

    def send_quick_reply(reply: str) -> None:
        send_message(reply)

    tk.Button(buttons_frame, text="Yes", command=lambda: send_quick_reply("Yes")).pack(
        side=tk.LEFT, padx=5
    )
    tk.Button(buttons_frame, text="No", command=lambda: send_quick_reply("No")).pack(
        side=tk.LEFT, padx=5
    )
    tk.Button(
        buttons_frame,
        text="Generate tailored CV",
        command=lambda: send_quick_reply("Please draft a tailored CV using the best matches."),
    ).pack(side=tk.LEFT, padx=5)

    root.mainloop()


def process_job_description(config: PipelineConfig, jd_path: Path) -> None:
    """Full pipeline triggered when a new JD file arrives."""
    LOG.info("Processing new JD: %s", jd_path.name)
    # Wait briefly to ensure file copy/transfer is complete.
    time.sleep(1.0)

    jd_text_raw = load_text(jd_path)
    jd_text = clean(jd_text_raw)
    keywords = extract_keywords(jd_text, config.top_keywords)
    update_skills_yaml(config.skills_path, keywords)

    ranking = rank_cvs(
        jd_path=jd_path,
        cv_dir=config.cv_dir,
        out_path=config.report_path,
        skills_path=config.skills_path,
        min_length=config.min_length,
    )

    prompt = build_initial_prompt(jd_text_raw, keywords, ranking)
    conversation = [
        {
            "role": "system",
            "content": (
                "You are an expert career consultant. "
                "Review CV rankings and advise on the best match and tailoring strategy."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        initial_reply = call_chatgpt(conversation, model=config.chat_model)
    except Exception as exc:
        LOG.error("ChatGPT call failed: %s", exc)
        return
    conversation.append({"role": "assistant", "content": initial_reply})
    show_chat_window(conversation, model=config.chat_model)


class JDFileHandler(FileSystemEventHandler):
    """Watchdog handler that pushes new JD files onto a queue."""

    def __init__(self, config: PipelineConfig, queue: Queue[Path]) -> None:
        super().__init__()
        self.config = config
        self.queue = queue

    def on_created(self, event):  # pragma: no cover - filesystem side effects
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in EXT_READERS:
            LOG.info("Ignoring unsupported JD file type: %s", path.name)
            return
        LOG.info("Detected new JD file: %s", path.name)
        self.queue.put(path)


def worker(config: PipelineConfig, queue: Queue[Path]) -> None:
    """Background thread that processes JD files sequentially."""
    while True:  # pragma: no cover - long running loop
        jd_path = queue.get()
        try:
            process_job_description(config, jd_path)
        except Exception as exc:
            LOG.exception("Failed to process %s: %s", jd_path, exc)
        finally:
            queue.task_done()


def main() -> None:
    config = PipelineConfig()
    ensure_directories(config)

    queue: Queue[Path] = Queue()
    observer = Observer()
    handler = JDFileHandler(config, queue)
    observer.schedule(handler, str(config.jd_dir), recursive=False)
    observer.start()
    LOG.info("Watching JD directory: %s", config.jd_dir)

    worker_thread = threading.Thread(target=worker, args=(config, queue), daemon=True)
    worker_thread.start()

    try:
        while True:  # Keep main thread alive for the watcher
            time.sleep(1.0)
    except KeyboardInterrupt:  # pragma: no cover - manual shutdown
        LOG.info("Stopping observer...")
        observer.stop()
    observer.join()


if __name__ == "__main__":  # pragma: no cover
    main()
