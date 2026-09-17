# dmri_project — Setup for group work

Recommended minimal workflow for all team members (portable and simple):

1. Create a local virtual environment and install pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Copy the example env and edit values locally:

```bash
cp .env.example .env
# edit .env as needed (do not commit .env)
```

3. Run the project from the activated venv:

```bash
source .venv/bin/activate
python dmri_project.py
```

Notes:
- We ignore `.venv` and `.env` in `.gitignore`.
- If a teammate prefers `conda` or `poetry`, document that alternative here.
- If installation fails due to disk space, free space and retry (remove `.venv` first).
