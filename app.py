from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4

import logging
import os
import tempfile
import warnings

import pandas as pd
import plotly.express as px
import streamlit as st
try:
    import sentry_sdk
except ImportError:  # pragma: no cover - optional dependency.
    sentry_sdk = None

from match_cvs import (
    EXT_READERS,
    clean,
    load_skills,
    load_text,
    skill_coverage,
    limit_text_to_pages,
    tfidf_similarity,
)


logger = logging.getLogger("cv_alignment_studio")


METRICS_HELP = (
    "There isn't a single 'true match' cut-off because the two numbers capture different signals:\n\n"
    "**Similarity (TF-IDF %)** - how close the overall language is to the JD. Values above ~15 usually show "
    "meaningful overlap; 40+ is rare unless the CV mirrors the JD.\n\n"
    "**Skill coverage %** - how many of the tracked skills are explicitly named. High coverage (60-80%) "
    "means most keywords were found.\n\n"
    "Use them together:\n"
    "- High similarity + high coverage -> strong match.\n"
    "- High coverage but low similarity -> lists the right skills, wording differs; review manually.\n"
    "- High similarity but low coverage -> phrasing matches but skills are missing; update the skill list or the CV.\n"
    "- Both low -> likely a weak match.\n\n"
    "You can sort by a blended score (e.g., 60% similarity + 40% coverage) to get a single ranking."
)


def apply_base_theme() -> None:
    st.markdown(
        """
        <style>
            [data-testid="stAppViewContainer"] {
                background: linear-gradient(180deg, #eff3ff 0%, #f9fbff 45%, #eef4ff 100%);
                padding-top: 0 !important;
            }
            .hero {
                text-align: left;
                padding: 2.2rem 0 1rem 0;
            }
            .hero-title {
                font-size: 2.4rem;
                font-weight: 700;
                color: #1f3c88;
                margin-bottom: 0.25rem;
            }
            .hero-subtitle {
                font-size: 1.15rem;
                font-weight: 500;
                color: #2d4379;
                margin-bottom: 0.35rem;
            }
            .hero-caption {
                font-size: 0.95rem;
                color: #3a4a82;
                opacity: 0.85;
            }
            .glass-card {
                background: rgba(255, 255, 255, 0.88);
                backdrop-filter: blur(15px);
                border-radius: 22px;
                padding: 1.5rem 1.75rem;
                box-shadow: 0 12px 32px rgba(31, 60, 136, 0.12);
                border: 1px solid rgba(185, 201, 255, 0.55);
                margin-bottom: 1.2rem;
            }
            .section-title {
                font-size: 1.05rem;
                font-weight: 600;
                color: #273a7a;
                margin-bottom: 0.6rem;
            }
            .tip-list {
                padding-left: 1.1rem;
                color: #33407d;
                font-size: 0.92rem;
                line-height: 1.45;
            }
            .meta-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.35rem 0.65rem;
                border-radius: 999px;
                background: rgba(104, 125, 238, 0.12);
                color: #5053a6;
                font-size: 0.82rem;
                margin-top: 0.4rem;
            }
            .meta-badge svg {
                width: 14px;
                height: 14px;
            }
            div[data-testid="stFileUploader"] > div:first-child {
                border: 2px dashed rgba(111, 139, 255, 0.55);
                background: rgba(249, 250, 255, 0.92);
                border-radius: 18px;
                padding: 1.4rem 1.2rem;
            }
            div[data-testid="stFileUploader"] label {
                font-weight: 600;
                color: #2b3c7e;
                font-size: 0.95rem;
            }
            div[data-testid="stFileUploaderDropzone"] p {
                color: #4551a3;
                font-weight: 500;
            }
            div[data-testid="stFileUploaderDropzone"] small {
                color: #6673c3;
            }
            div[data-testid="stFileUploader"] button {
                border-radius: 999px;
                background: linear-gradient(120deg, #7a8dff 0%, #5571ff 100%);
                color: #fff;
                font-weight: 600;
            }
            div[data-testid="stTextInput"] input {
                border-radius: 14px;
                border: 1.5px solid rgba(156, 175, 255, 0.55);
                background: rgba(255, 255, 255, 0.92);
                font-size: 0.94rem;
            }
            div[data-testid="stNumberInput"] input {
                border-radius: 14px;
                border: 1.5px solid rgba(156, 175, 255, 0.55);
                background: rgba(255, 255, 255, 0.92);
                font-size: 0.94rem;
            }
            div[data-testid="stNumberInput"] button {
                background: transparent;
                color: #5160b3;
            }
            div[data-testid="stMarkdown"].quick-tips {
                background: rgba(255, 255, 255, 0.8);
                border-radius: 18px;
                padding: 1.1rem 1.3rem;
                border: 1px solid rgba(207, 217, 255, 0.7);
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7);
            }
            div[data-testid="stButton"][id*="run_btn"] button {
                width: 100%;
                border-radius: 999px;
                background: linear-gradient(120deg, #5780ff 0%, #315bff 100%);
                box-shadow: 0 10px 20px rgba(72, 108, 255, 0.25);
                color: #fff;
                font-weight: 600;
                padding: 0.6rem 0;
            }
            div[data-testid="stButton"][id*="stop_btn"] button {
                width: 100%;
                border-radius: 999px;
                background: linear-gradient(120deg, #ff6b6b 0%, #e63946 100%);
                box-shadow: 0 10px 20px rgba(230, 57, 70, 0.25);
                color: #fff;
                font-weight: 600;
                padding: 0.6rem 0;
            }
            div[data-testid="stButton"][id*="stop_btn"] button:hover {
                filter: brightness(1.02);
            }
            div[data-testid="stPopover"] * {
                color: #1f2755 !important;
            }
            div[data-testid="stPopover"] {
                border-radius: 18px;
                border: 1px solid rgba(164, 182, 255, 0.6);
                box-shadow: 0 14px 36px rgba(37, 61, 110, 0.22);
            }
            div[data-testid="stDataFrame"] {
                border-radius: 18px;
                overflow: hidden;
                border: 1px solid rgba(182, 197, 255, 0.6);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_bytes(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            if unit == "B":
                return f"{int(num)}{unit}"
            return f"{num:.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}TB"


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">CV Alignment Studio (Local)</div>
            <div class="hero-subtitle">Match. Compare. Tailor &mdash; Turn job specs into standout CVs in minutes.</div>
            <div class="hero-caption">Rank CVs against a local Job Description using TF-IDF similarity and skill coverage.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quick_tips() -> None:
    st.markdown(
        """
        <div class="quick-tips">
            <div class="section-title" style="margin-bottom:0.35rem;">Quick tips</div>
            <ul class="tip-list">
                <li>Keep only candidate CVs in the folder to speed up processing.</li>
                <li>Use the skills YAML to tune what "coverage" means for each role.</li>
                <li>Lower the minimum character threshold if shorter CVs are being skipped.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_matching(
    jd_path: Path,
    cv_dir: Path,
    out_path: Path,
    skills_path: Path,
    min_length: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    if not jd_path.exists():
        raise FileNotFoundError(f"JD not found: {jd_path}")
    if not cv_dir.exists() or not cv_dir.is_dir():
        raise FileNotFoundError(f"CV folder not found: {cv_dir}")
    if not skills_path.exists():
        raise FileNotFoundError(f"Skills YAML not found: {skills_path}")

    jd_text_raw = load_text(jd_path)
    if not jd_text_raw.strip():
        raise ValueError("JD file appears to be empty after extraction.")
    jd_text_raw, jd_trimmed = limit_text_to_pages(jd_text_raw)
    if jd_trimmed:
        st.info("Job description truncated to the first two pages for analysis.")
    jd_text = clean(jd_text_raw)

    skills = load_skills(skills_path)

    cv_files = [p for p in cv_dir.iterdir() if p.suffix.lower() in EXT_READERS]
    if not cv_files:
        raise ValueError("No supported CV files found in the folder.")

    texts_raw, kept_files = [], []
    for p in cv_files:
        try:
            text = load_text(p)
        except Exception as exc:
            st.warning(f"Failed to read {p.name}: {exc}")
            continue
        if len(text) < min_length:
            st.info(f"Skipping {p.name}: too short ({len(text)} chars)")
            continue
        texts_raw.append(text)
        kept_files.append(p)

    if not texts_raw:
        raise ValueError(
            "No CVs passed the minimum length filter. Lower the minimum length if needed."
        )

    texts = [clean(t) for t in texts_raw]
    similarities = tfidf_similarity(jd_text, texts)

    rows = []
    for path, text, similarity in zip(kept_files, texts, similarities):
        matched, missing = skill_coverage(text, skills)
        coverage = 100.0 * len(matched) / max(1, len(skills))
        rows.append(
            {
                "cv_file": path.name,
                "similarity": round(similarity, 2),
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

    top_missing = Counter()
    for missing in df.head(3)["missing_skills"].tolist():
        top_missing.update(
            [skill.strip() for skill in missing.split(",") if skill.strip()]
        )

    missing_series = pd.Series(top_missing, dtype=int)
    missing_series = missing_series.sort_values(ascending=False)

    return df, missing_series


def main() -> None:
    st.set_page_config(page_title="CV Alignment Studio", layout="wide")
    apply_base_theme()
    render_header()

    default_jd_dir = Path.cwd() / "JD"
    default_cv_dir = Path.home() / "Downloads" / "CV"

    defaults = {
        "jd_path_input": str(default_jd_dir),
        "cv_dir_input": str(default_cv_dir),
        "skills_path_input": str(Path.cwd() / "skills.yaml"),
        "out_path_input": str(Path.cwd() / "report.csv"),
        "min_len_input": 150,
    }
    for key, default in defaults.items():
        st.session_state.setdefault(key, default)

    st.session_state.setdefault("uploaded_jd_meta", None)
    st.session_state.setdefault("last_dataframe", None)
    st.session_state.setdefault("last_missing_summary", None)
    st.session_state.setdefault("last_error", "")
    st.session_state.setdefault("last_report_path", "")
    st.session_state.setdefault("stop_requested", False)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload job description</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag and drop job file here",
        type=["pdf", "docx", "txt"],
        key="jd_file_uploader",
        help="Limit 200MB per file - PDF, DOCX, TXT",
        label_visibility="collapsed",
    )
    if uploaded_file:
        tmp_dir = Path(tempfile.gettempdir())
        safe_name = f"cvstudio_jd_{uuid4().hex}_{uploaded_file.name}"
        tmp_path = tmp_dir / safe_name
        tmp_path.write_bytes(uploaded_file.getbuffer())
        st.session_state["jd_path_input"] = str(tmp_path)
        st.session_state["uploaded_jd_meta"] = {
            "name": uploaded_file.name,
            "size": format_bytes(uploaded_file.size),
        }
        st.toast(f"JD uploaded: {uploaded_file.name}")

    if st.session_state["uploaded_jd_meta"]:
        meta = st.session_state["uploaded_jd_meta"]
        st.markdown(
            f"""
            <span class="meta-badge">
                <svg viewBox="0 0 24 24" fill="none">
                    <path d="M12 3L2 9l10 6 10-6-10-6zm0 7.5L4.5 9 12 4.5 19.5 9 12 10.5z" fill="#5468ff"/>
                </svg>
                {meta['name']} &bull; {meta['size']}
            </span>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    cols = st.columns(2, gap="large")
    with cols[0]:
        jd_path_str = st.text_input(
            "Job description file path (optional if uploaded)",
            key="jd_path_input",
            help="If you uploaded a file above this will point to its temporary copy.",
        )
        skills_path_str = st.text_input(
            "Skills YAML path",
            key="skills_path_input",
            help="List of skills tracked for coverage. Update to tune the metric.",
        )
    with cols[1]:
        cv_dir_str = st.text_input(
            "CV folder path",
            key="cv_dir_input",
            help="Folder containing CVs in PDF, DOCX, or TXT format.",
        )
        out_path_str = st.text_input(
            "Output CSV path",
            key="out_path_input",
            help="Results are saved here and can be downloaded below.",
        )

    control_cols = st.columns([3, 1, 1], gap="large")
    with control_cols[0]:
        min_length = st.number_input(
            "Minimum characters in CV",
            min_value=100,
            max_value=5000,
            step=50,
            key="min_len_input",
            help="Skip CVs shorter than this threshold.",
        )
    with control_cols[1]:
        run_clicked = st.button("Run matching", key="run_btn")
    with control_cols[2]:
        stop_clicked = st.button("Stop matching", key="stop_btn")
    st.markdown("</div>", unsafe_allow_html=True)

    render_quick_tips()

    if stop_clicked:
        st.session_state["stop_requested"] = True
        st.session_state["last_error"] = "Matching stopped. Adjust your inputs and click Run matching again."
        st.session_state["last_dataframe"] = None
        st.session_state["last_missing_summary"] = None
        st.session_state["last_report_path"] = ""

    if run_clicked:
        st.session_state["stop_requested"] = False
        try:
            dataframe, missing_summary = run_matching(
                jd_path=Path(jd_path_str).expanduser(),
                cv_dir=Path(cv_dir_str).expanduser(),
                out_path=Path(out_path_str).expanduser(),
                skills_path=Path(skills_path_str).expanduser(),
                min_length=int(min_length),
            )
        except Exception as exc:
            st.session_state["last_error"] = str(exc)
            st.session_state["last_dataframe"] = None
            st.session_state["last_missing_summary"] = None
            st.session_state["last_report_path"] = ""
        else:
            st.session_state["last_error"] = ""
            st.session_state["last_dataframe"] = dataframe
            st.session_state["last_missing_summary"] = missing_summary
            st.session_state["last_report_path"] = out_path_str

    last_error = st.session_state.get("last_error")
    last_dataframe: Optional[pd.DataFrame] = st.session_state.get("last_dataframe")
    last_missing: Optional[pd.Series] = st.session_state.get("last_missing_summary")
    last_report_path = st.session_state.get("last_report_path")

    if last_error:
        st.error(last_error)

    if last_dataframe is not None and not last_dataframe.empty:
        if last_report_path:
            st.success(f"Report saved to {last_report_path}")

        top_row = last_dataframe.iloc[0]
        st.markdown(
            f"**Top CV:** `{top_row['cv_file']}` &nbsp;&bull;&nbsp; Similarity {top_row['similarity']:.2f}% "
            f"| Skill coverage {top_row['skill_coverage_pct']:.1f}%",
            unsafe_allow_html=True,
        )
        if top_row["missing_skills"]:
            st.markdown(f"**Missing skills:** {top_row['missing_skills']}")

        display_df = last_dataframe.copy()
        display_df.insert(1, "Similarity %", display_df["similarity"].map(lambda v: f"{v:.2f}%"))
        display_df["Skill coverage %"] = display_df["skill_coverage_pct"].map(lambda v: f"{v:.1f}%")
        display_df = display_df[
            [
                "cv_file",
                "Similarity %",
                "Skill coverage %",
                "matched_skills",
                "missing_skills",
                "cv_chars",
            ]
        ].rename(
            columns={
                "cv_file": "CV file",
                "matched_skills": "Matched skills",
                "missing_skills": "Missing skills",
                "cv_chars": "CV characters",
            }
        )

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        popover = st.popover("? What do these metrics mean?")
        with popover:
            st.markdown(METRICS_HELP)

        if last_missing is not None and not last_missing.empty:
            st.subheader("Common missing skills across top CVs")
            missing_sorted = last_missing.sort_values(ascending=True)
            chart = px.bar(
                missing_sorted,
                x=missing_sorted.values,
                y=missing_sorted.index,
                orientation="h",
                labels={"x": "Frequency", "y": "Skill"},
                title=None,
            )
            chart.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(chart, width="stretch")

        csv_bytes = last_dataframe.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name=Path(last_report_path).name if last_report_path else "report.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="The keyword arguments have been deprecated and will be removed in a future release.",
    )

    logger = logging.getLogger("cv_alignment_studio")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if sentry_sdk and (dsn := os.getenv("SENTRY_DSN")):
        sentry_sdk.init(dsn=dsn)
        logger.info("Sentry monitoring initialised.")

    main()
