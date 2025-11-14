import sys
import os

# Ensure project root is on sys.path so tests can import `app` package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_toggle_cycle():
    # single_press should be a noop
    r = client.post('/api/single_press')
    assert r.status_code == 200
    assert r.json().get('action') == 'SINGLE_PRESS'

    # double_press should return an action and a song string
    r = client.post('/api/double_press')
    assert r.status_code == 200
    body = r.json()
    assert body.get('action') == 'DOUBLE_PRESS'
    assert isinstance(body.get('song'), str)

    # calling double_press again should also return a song
    r2 = client.post('/api/double_press')
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2.get('action') == 'DOUBLE_PRESS'
    assert isinstance(body2.get('song'), str)
