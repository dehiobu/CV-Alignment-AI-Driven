import textwrap
from pathlib import Path

import pandas as pd
import pytest

import match_cvs as mc
import app as app_module


def test_clean_normalises_whitespace_and_case():
    dirty = " Senior DATA Engineer \n\n\tExperience in PYTHON  "
    assert mc.clean(dirty) == " senior data engineer experience in python "


def test_skill_coverage_identifies_matches_and_missing():
    text = "python and sql with cloud experience"
    skills = ["python", "sql", "aws"]

    matched, missing = mc.skill_coverage(text, skills)

    assert matched == ["python", "sql"]
    assert missing == ["aws"]


def test_limit_text_to_pages_truncates_long_text():
    base = "lorem ipsum " * 1000
    truncated, was_trimmed = mc.limit_text_to_pages(base, max_pages=1, chars_per_page=1000)

    assert was_trimmed is True
    assert len(truncated) <= 1000
    # ensure we do not cut mid-word badly
    assert truncated.endswith(" ") or truncated.endswith("m")


def test_tfidf_similarity_ranks_closer_document_higher():
    jd = "python data engineering cloud pipelines automation"
    close_cv = "python automation of data pipelines on cloud"
    far_cv = "front end design marketing copywriting graphics"

    scores = mc.tfidf_similarity(mc.clean(jd), [mc.clean(close_cv), mc.clean(far_cv)])

    assert scores[0] > scores[1]


def test_load_skills_reads_yaml(tmp_path: Path):
    yaml_content = textwrap.dedent(
        """
        engineering:
          - python
          - sql
        cloud:
          - aws
        """
    )
    skills_file = tmp_path / "skills.yml"
    skills_file.write_text(yaml_content)

    skills = mc.load_skills(skills_file)

    assert skills == ["aws", "python", "sql"]


def test_run_matching_produces_dataframe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    jd_path = tmp_path / "jd.txt"
    jd_path.write_text("python developer cloud data pipelines")

    cv_dir = tmp_path / "cvs"
    cv_dir.mkdir()
    (cv_dir / "cv1.txt").write_text("python developer with cloud skills and pipelines experience")
    (cv_dir / "cv2.txt").write_text("graphic designer portfolio management")  # filtered by similarity

    out_path = tmp_path / "report.csv"
    skills_path = tmp_path / "skills.yml"
    skills_path.write_text(
        textwrap.dedent(
            """
            engineering:
              - python
              - cloud
              - pipelines
            """
        )
    )

    # Streamlit functions emit UI warnings. Patch to no-op for tests.
    monkeypatch.setattr(app_module.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_module.st, "info", lambda *args, **kwargs: None)

    df, missing = app_module.run_matching(
        jd_path=jd_path,
        cv_dir=cv_dir,
        out_path=out_path,
        skills_path=skills_path,
        min_length=10,
    )

    assert isinstance(df, pd.DataFrame)
    assert "cv_file" in df.columns
    assert out_path.exists()
    assert missing.empty or isinstance(missing, pd.Series)
