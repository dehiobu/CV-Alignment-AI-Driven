# CV Alignment Studio

AI-assisted platform for ranking CVs against local job descriptions. The Streamlit UI delivers a polished, SaaS-ready experience: drag in a JD, point at a CV folder, and instantly compare similarity plus skill coverage to surface the best-fitting candidates.

---

## Highlights

- **Drag-and-drop JD upload** with automatic fallback to manual path entry and persistent metadata badges.
- **Side-by-side controls** for CV folder, skills taxonomy, minimum character threshold, and output path – all pre-filled with sensible defaults.
- **Interactive ranking table** summarising similarity %, skill coverage %, matched/missing keywords, and character counts, with CSV export in one click.
- **Quick metrics guidance** delivered via inline popover so recruiters understand how to interpret scores.
- **Horizontal missing-skill chart** to spot collective gaps across the strongest CVs.
- **"Stop matching" safety** for quick aborts when the wrong directory is selected.
- **Automation hooks** (`automation.py`) to monitor a JD drop folder, auto-refresh `skills.yaml`, call OpenAI for recommendations, and keep humans in the loop via a Tkinter chat window.
- **CLI parity** (`match_cvs.py`) for scripting and batch runs without the UI.

---

## Architecture Overview

| Component        | Purpose                                                                 |
|------------------|-------------------------------------------------------------------------|
| `app.py`         | Streamlit frontend; orchestrates uploads, input validation, matching UI |
| `match_cvs.py`   | Core TF-IDF and skill coverage logic used by both UI and CLI            |
| `automation.py`  | Optional watcher + OpenAI automation pipeline                           |
| `skills.yaml`    | Default skill taxonomy grouped by capability                            |
| `scripts/`       | PowerShell helpers for setup and launching the app                      |
| `tests/`         | Pytest suite covering cleaning, skills loading, and the matching flow   |
| `.github/workflows/ci.yml` | Ready-to-run GitHub Actions pipeline for linting/tests        |

The matching engine prefers scikit-learn’s TF-IDF vectoriser when installed, but gracefully falls back to a lightweight pure-Python implementation so the app remains usable in constrained environments.

---

## Prerequisites

- Windows 10/11 (PowerShell scripts are provided), macOS, or Linux.
- Python 3.11 (3.10–3.13 supported; 3.11 is the project default).
- Optional tooling: Git, Docker, and a modern browser for Streamlit.
- For automation/OpenAI features, set `OPENAI_API_KEY` and ensure outbound network access.

---

## Getting Started

### 1. Automated setup (recommended on Windows)
```powershell
# Install dependencies into .venv and fetch dev tooling
pwsh scripts/setup.ps1 -Dev

# Launch the Streamlit UI
pwsh scripts/run-app.ps1
```

### 2. Manual virtual environment
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

### 3. Run the UI
```powershell
streamlit run app.py
```
The app opens at `http://localhost:8501`. Upload a JD or provide a file path, pick your CV folder, adjust thresholds, and click **Run matching**. Download the generated CSV or inspect missing skill trends directly in the browser.

### 4. Command-line batch matching
```bash
python match_cvs.py \
  --jd JD/JD_Unstructured\ Business\ Data\ Consultant\ \(Onshore\)\ \(2\).docx \
  --cvs examples/CVs \
  --out report.csv
```

### 5. Automation pipeline (optional)
```bash
python automation.py
```
The script watches the `JD/` directory for new files, updates `skills.yaml`, runs the matcher, and launches a Tkinter chat dialogue seeded with OpenAI guidance. Requires `OPENAI_API_KEY` and (optionally) Sentry configuration.

---

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_ENV` | Environment label for logs/Sentry | `development` |
| `APP_LOG_LEVEL` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |
| `SENTRY_DSN` | Enables Sentry monitoring when provided | unset |
| `SENTRY_TRACES_SAMPLE_RATE` | Capture percentage for tracing | `0.05` (if set) |
| `SENTRY_PROFILES_SAMPLE_RATE` | Capture percentage for profiling | `0.0` |
| `OPENAI_API_KEY` | Used by `automation.py` to call OpenAI chat completions | unset |

All environment variables can be placed in a `.streamlit/secrets.toml` or exported before launching the app.

---

## Testing & Quality Gates

```bash
ruff check .                # Lint
mypy app.py match_cvs.py    # Type checking
pytest                      # Unit tests
```

GitHub Actions (`.github/workflows/ci.yml`) runs these same steps on every push to `main` or PR branch.

---

## Deployment Options

### Docker
```bash
docker build -t cv-alignment-studio:latest .
docker run --rm -p 8501:8501 cv-alignment-studio:latest
```

### Streamlit Community Cloud
1. Push this repository to GitHub (already done: `dehiobu/CV-Alignment-AI-Driven`).
2. Sign into Streamlit Cloud, click **New app**, and point to `main` / `app.py`.
3. Add required secrets (`OPENAI_API_KEY`, Sentry DSN, etc.) in the workspace settings.

### Production SaaS considerations
- Place Streamlit behind HTTPS via Cloudflare, Nginx, or your preferred load balancer.
- Mount persistent storage/S3 for output CSVs if end-users export frequently.
- Integrate authentication (e.g., Auth0, Cognito) for customer-tenanted access.
- Add billing (Stripe, Paddle) and rate limiting as part of SaaS hardening roadmap (see below).

---

## Sample Data

- `JD/` contains an example job description (Business Data Consultant).
- `examples/CVs/` hosts sample CV snippets to test scoring quickly.
- `skills.yaml` defines skills grouped by domain; edit or extend to align with your industry or role taxonomy.

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'sklearn'` | Optional dependency missing | Install `scikit-learn` (or rely on built-in fallback TF-IDF) |
| Streamlit warning: `The keyword arguments have been deprecated...` | Old Plotly config flags | Already resolved – ensure you're on the latest `main` |
| Uploaded JD keeps showing cached text | Streamlit caching temp file | Use **Stop matching** then re-run; temporary path updates automatically |
| Automation script crashes on Tkinter import | Running in headless environment | Install a desktop-capable Python or disable automation module |
| Long path names on Windows | Path exceeds 260 characters | Enable long-path support or shorten directory names |

---

## Roadmap Ideas

1. Multi-tenant authentication, granular RBAC, and audit logging.
2. Weighted scoring configuration per customer/role.
3. GPT-based hybrid CV builder tuned to job description language.
4. REST API/webhooks for automated JD intake and report retrieval.
5. Real-time collaboration (shared comments, recruiter notes).
6. Billing stack integration for full SaaS offering.

---

## Contributing

1. Fork the repo and create a feature branch.
2. Run `ruff`, `mypy`, and `pytest` before submitting PRs.
3. Follow conventional commit messages when possible (`feat:`, `fix:`, etc.).
4. Document UI changes with screenshots in the PR description.

---

## License & Support

This project is currently proprietary; contact **support@cv-alignment-studio.com** for licensing discussions, enterprise support, or implementation services.
