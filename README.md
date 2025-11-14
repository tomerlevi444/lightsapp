# Lights App (FastAPI)

Simple demo app that shows a button in the browser to toggle lights on/off. When the button is pressed the page shows `LIGHTS ON` or `LIGHTS OFF` and the button background toggles to red when on.

Run locally (macOS / zsh):

1. Create a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app with uvicorn (from project root):

```bash
uvicorn app.main:app --reload
```

4. Open the frontend at http://127.0.0.1:8000/

Run tests:

```bash
pytest -q
```
