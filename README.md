# CV Alignment Studio

Streamlit-powered platform for ranking CVs against job descriptions using TF-IDF similarity and skill coverage analytics. Designed for SaaS deployment—everything runs server-side so customers only need a browser.

---

## Feature Snapshot

- Drag-and-drop JD upload and bulk CV directory ingestion (PDF/DOCX/TXT).
- TF-IDF similarity scoring, skill coverage percentage, and explainable matched/missing skill breakdowns.
- Sentry-ready observability, structured logging, and graceful error handling.
- Horizontal skill-gap visualisations with expandable full-screen charts.
- Automated CSV export and optional OpenAI integration hook (via existing automation.py).

---

## Quickstart

1. **Automated setup & run (Windows/PowerShell)**
   ```powershell
   # prepare environment (adds .venv and installs deps)
   pwsh scripts/setup.ps1 -Dev

   # launch Streamlit UI (assumes setup already run)
   pwsh scripts/run-app.ps1
   ```

2. **Manual setup (if you prefer explicit commands)**
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements-dev.txt
   ```

3. **Run the Streamlit UI manually**
   ```powershell
   streamlit run app.py
   ```

4. **Run the CLI batch matcher (optional)**
   ```powershell
   python match_cvs.py --jd path/to/JD.docx --cvs path/to/cv_folder --out report.csv
   ```

5. **Execute quality gates**
   ```powershell
   ruff check .
   mypy app.py match_cvs.py
   pytest
   ```

---

## Project Layout

```
app.py                # Streamlit frontend
match_cvs.py          # Core matching logic (imported by the app + CLI)
automation.py         # Optional automation hooks / integrations
skills.yaml           # Default skill taxonomy
tests/                # Pytest suite
requirements*.txt     # Runtime & dev dependencies
Dockerfile            # Production container
pyproject.toml        # Packaging & toolchain config
```

---

## Environment Configuration

The app reads runtime settings from environment variables. Recommended settings for production hosting:

| Variable | Purpose | Example |
|----------|---------|---------|
| `APP_ENV` | Environment label for logs/Sentry | `production` |
| `APP_LOG_LEVEL` | Logging verbosity (`DEBUG`, `INFO`, …) | `INFO` |
| `SENTRY_DSN` | Enables Sentry error tracking when present | `https://...@sentry.io/...` |
| `SENTRY_TRACES_SAMPLE_RATE` | Trace sampling (0-1) | `0.1` |
| `SENTRY_PROFILES_SAMPLE_RATE` | Profile sampling (0-1) | `0.0` |

---

## Docker Deployment (SaaS-ready)

1. Build & tag the container:
   ```bash
   docker build -t cv-alignment-studio:latest .
   ```
2. Run locally:
   ```bash
   docker run --rm -p 8501:8501 \
     -e APP_ENV=production \
     -e APP_LOG_LEVEL=INFO \
     cv-alignment-studio:latest
   ```
3. Deploy to your orchestrator (ECS, GKE, AKS, etc.) with:
   - Secrets manager for Sentry/OpenAI/API keys.
   - Persistent volume or S3 for generated reports.
   - HTTPS termination (ALB/Ingress + certificate).

---

## QA & Release Workflow

1. Format & lint with Ruff.
2. Type-check with mypy.
3. Run the unit/integration test suite (`pytest`).
4. Build Docker image and execute smoke tests.
5. Publish artefacts via CI (GitHub Actions workflow recommended).
6. Tag release and deploy.

See `pyproject.toml` for configured tool versions and `requirements-dev.txt` for the dev stack.

---

## Roadmap Ideas

- Multi-tenant authentication & subscription billing (Stripe, Paddle).
- Webhook/API endpoints for automated JD/CV ingestion.
- Embedding-based semantic similarity as an optional upgrade.
- Team workspaces with shared reports and audit logs.

---

## Support

For enterprise licences or integration services, contact **support@example.com**.

© 2025 CV Alignment Studio. All rights reserved.
