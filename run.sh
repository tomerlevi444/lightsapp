python3 -m venv .venv
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info --no-access-log
