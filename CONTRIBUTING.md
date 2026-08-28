# Contributing

Thanks for your interest in improving the AI Research Assistant! This is a
portfolio project, but issues and pull requests are welcome.

## Local setup

```bash
# 1. Clone the repo
git clone https://github.com/sumitjhadev/ai-research-assistant.git
cd ai-research-assistant

# 2. Create a virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies (dev extras include pytest + ruff)
pip install -r requirements-dev.txt

# 4. Configure your environment
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY (get a free key at
# https://aistudio.google.com/app/apikey)

# 5. Ingest a corpus (defaults to "retrieval augmented generation")
python backend/ingest.py --query "retrieval augmented generation" --max-results 50

# 6. Build the vector store
python backend/vectorstore.py build

# 7. Run the backend
uvicorn backend.api:app --reload --port 8000

# 8. In a second terminal, run the frontend
streamlit run app.py
```

## Running tests

```bash
pytest -v --cov=backend
```

All Gemini API calls are mocked in tests — you do not need a real API key to
run the test suite or contribute.

## Linting

```bash
ruff check .
ruff format .
```

## Pull requests

1. Fork the repo and create a feature branch.
2. Keep functions short, typed, and documented (docstrings + type hints).
3. Add or update tests for any behavior change.
4. Make sure `pytest` and `ruff check .` both pass before opening a PR.
5. Describe what you changed and why in the PR description.

## Code style

- Python 3.11+, type hints on every function signature.
- Structured logging via `backend.config.get_logger`, never `print()`.
- No hardcoded secrets — everything sensitive comes from environment
  variables loaded via `.env` (see `.env.example`).
