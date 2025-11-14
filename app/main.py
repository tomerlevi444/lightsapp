from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import logging
import httpx
import random
import threading
import json
from typing import Optional
import subprocess
import sys
import shutil
import glob
from pathlib import Path

# Configure basic logging so INFO/ERROR messages are visible on the console by default.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

app = FastAPI()

# Songs list (90s) - shuffled once at startup
songs = [
    "Nirvana - Smells Like Teen Spirit",
    "Oasis - Wonderwall",
    "TLC - No Scrubs",
    "R.E.M. - Losing My Religion",
    "Spice Girls - Wannabe",
]
# If a saved state exists, we'll load it at startup; otherwise shuffle once now.
random.shuffle(songs)
# index of the currently playing song in the songs list
current_song_index = 0
# lock to protect current_song_index for concurrent access
songs_lock = threading.Lock()

# Playback process (non-blocking) and lock so we can stop previous playback
playback_process: subprocess.Popen | None = None
playback_lock = threading.Lock()
# generation counter to cancel watchers when playback is stopped
playback_generation = 0

# Power control endpoints
POWER_HOSTS = os.getenv("POWER_HOSTS", "192.168.1.104")
POWER_URLS = [f"http://{h}/cm?cmnd=Power%20On" for h in POWER_HOSTS.split(",")]  # default ON urls
POWER_OFF_URLS = [u.replace("Power%20On", "Power%20Off") for u in POWER_URLS]

# File used to persist state across restarts. Located in the repo/workdir so it's
# easy to inspect during development. If you prefer another location, set
# STATE_FILE env var to override.
STATE_FILE = os.getenv("STATE_FILE", os.path.join(os.getcwd(), "state.json"))


def load_state() -> Optional[dict]:
    """Load persisted state from STATE_FILE if it exists.

    Returns the parsed dict or None on error / missing file.
    """
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        logger.exception("Failed to load state from %s", STATE_FILE)
        return None


def save_state() -> None:
    """Persist current songs list and index to STATE_FILE atomically."""
    try:
        payload = {"songs": songs, "current_index": current_song_index}
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, STATE_FILE)
    except Exception:
        logger.exception("Failed to save state to %s", STATE_FILE)


def _which_player() -> Optional[str]:
    """Return the first available playback command on the system or None.

    Preference: macOS 'afplay', then 'ffplay', then 'mpg123'. On Linux prefer
    'mpg123' or 'ffplay'."""
    candidates = []
    if sys.platform == "darwin":
        candidates = ["afplay", "ffplay", "mpg123"]
    elif sys.platform.startswith("linux"):
        candidates = ["mpg123", "ffplay", "aplay"]
    else:
        candidates = ["ffplay", "mpg123", "afplay"]

    for cmd in candidates:
        if shutil.which(cmd):
            return cmd
    return None


def _stop_playback():
    """Stop currently playing process if any (best-effort)."""
    global playback_process
    with playback_lock:
        if playback_process is not None:
            try:
                if playback_process.poll() is None:
                    playback_process.terminate()
            except Exception:
                logger.exception("Failed to terminate existing playback process")
            finally:
                playback_process = None
                # bump generation to cancel any pending end handlers
                global playback_generation
                playback_generation += 1


def _start_playback(file_path: str) -> dict:
    """Start playback of file_path in a non-blocking subprocess.

    Returns a small dict describing the attempt.
    """
    global playback_process
    player = _which_player()
    if not player:
        msg = "no playback command available (tried afplay/ffplay/mpg123)"
        logger.info("Playback skipped: %s", msg)
        return {"played": False, "path": file_path, "error": msg}

    # Build args depending on player
    if player == "afplay":
        args = [player, file_path]
    elif player == "ffplay":
        args = [player, "-nodisp", "-autoexit", "-loglevel", "quiet", file_path]
    elif player == "mpg123":
        args = [player, file_path]
    elif player == "aplay":
        args = [player, file_path]
    else:
        args = [player, file_path]

    try:
        # Stop previous and start new playback
        _stop_playback()
        p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with playback_lock:
            playback_process = p
            # advance generation so old watchers won't run
            global playback_generation
            playback_generation += 1
            my_generation = playback_generation

        logger.info("Started playback %s with %s (gen=%s)", file_path, player, my_generation)

        # Start a watcher thread that waits for process exit and then triggers
        # the post-playback actions (calls power endpoints). It only runs if
        # the generation still matches (so cancelled/stopped play won't fire).
        def _watcher(proc, gen):
            try:
                proc.wait()
            except Exception:
                logger.exception("Watcher error waiting for playback process")
            finally:
                with playback_lock:
                    current_gen = playback_generation
                logger.info("Playback process exited (gen=%s current_gen=%s)", gen, current_gen)
                if gen == current_gen:
                    # Only call power endpoints if this playback wasn't cancelled
                    logger.info("Playback finished; calling power-on endpoints")
                    # call in background so watcher returns quickly
                    threading.Thread(target=_call_power_endpoints, args=("on",), daemon=True).start()

        threading.Thread(target=_watcher, args=(p, my_generation), daemon=True).start()
        return {"played": True, "path": file_path, "player": player}
    except Exception as exc:
        logger.exception("Failed to start playback for %s", file_path)
        return {"played": False, "path": file_path, "error": str(exc)}


def play_song(song_name: str) -> dict:
    """Find an mp3 for the given song_name under the songs directory and play it.

    song_name is the display name like "Nirvana - Smells Like Teen Spirit" and
    files are expected to be named like "Nirvana - Smells Like Teen Spirit.mp3".
    Returns the dict returned by _start_playback or an error dict.
    """
    songs_dir = os.getenv("SONGS_DIR", os.path.join(os.getcwd(), "songs"))
    # First try exact filename
    exact = os.path.join(songs_dir, f"{song_name}.mp3")
    if os.path.exists(exact):
        return _start_playback(exact)

    msg = f"song mp3 not found for '{song_name}' in {songs_dir}"
    logger.info(msg)
    return {"played": False, "path": None, "error": msg}


def _call_power_endpoints(state: str) -> list:
    """Call the configured power endpoints.

    state should be 'on' or 'off'. Returns list of results from _post_forward.
    Runs synchronously; callers may invoke this in a background thread.
    """
    results = []
    urls = POWER_URLS if state.lower() == "on" else POWER_OFF_URLS
    for u in urls:
        try:
            # These Tasmota-style /cm?cmnd=Power... endpoints expect GET requests.
            resp = httpx.get(u, timeout=3.0)
            status = resp.status_code
            forwarded = 200 <= status < 300
            # Log non-successful responses but continue to the next endpoint.
            if not forwarded:
                logger.error("Power endpoint %s returned non-2xx status: %s", u, status)
            results.append({"url": u, "forwarded": forwarded, "status": status, "error": None})
        except Exception as exc:
            # Log the exception for visibility and continue to next endpoint.
            logger.exception("Error calling power endpoint %s", u)
            results.append({"url": u, "forwarded": False, "status": None, "error": str(exc)})
    return results


@app.on_event("startup")
def startup_event():
    """Set initial song at server startup (plays first song)."""
    # Ensure we try to power everything off on startup. Run in a background
    # thread so startup isn't blocked by slow/unavailable endpoints.
    try:
        threading.Thread(target=_call_power_endpoints, args=("off",), daemon=True).start()
    except Exception:
        # Protect startup from any unexpected threading issues; log and continue.
        logger.exception("Failed to start background power-off thread on startup")
    global current_song_index
    # Try to load persisted state if available. If loading fails, fall back to
    # the default shuffled songs list and start at index 0.
    state = load_state()
    with songs_lock:
        if state and isinstance(state, dict):
            loaded_songs = state.get("songs")
            loaded_index = state.get("current_index")
            if isinstance(loaded_songs, list) and loaded_songs:
                # Replace the songs list with persisted order and validate index.
                songs.clear()
                songs.extend(loaded_songs)
                if isinstance(loaded_index, int) and 0 <= loaded_index < len(songs):
                    current_song_index = loaded_index
                else:
                    current_song_index = 0
                logger.info("Loaded persisted state: playing initial song: %s", songs[current_song_index])
                return

        # Default fallback: ensure current index is within bounds and log it.
        if songs:
            current_song_index = 0
            logger.info("Startup: playing initial song: %s", songs[current_song_index])



@app.on_event("shutdown")
def shutdown_event():
    """Persist state on shutdown so restarts resume where we left off."""
    # save_state handles its own exceptions and logs failures
    save_state()





def _post_forward(url: str, json_payload: dict | None = None, timeout: float = 3.0) -> dict:
    """Helper to POST to an external URL and return a small result dict.

    Returns: { forwarded: bool, status: int|null, error: str|null }
    """
    try:
        resp = httpx.post(url, json=json_payload or {}, timeout=timeout)
        status = resp.status_code
        forwarded = 200 <= status < 300
        return {"forwarded": forwarded, "status": status, "error": None}
    except Exception as exc:  # pragma: no cover - network behavior
        # Log the full exception with stack trace so it's visible in server logs.
        logger.exception("Failed to POST to %s", url)
        # No additional mirrored logger; module logger already logs exception with stack trace.
        return {"forwarded": False, "status": None, "error": str(exc)}


@app.post("/api/single_press")
def single_press():
    """SINGLE_PRESS: play first song and call power-on endpoints.

    Behavior:
    - sets the current song to the first in the list (index 0)
    - persists state
    - triggers power-on endpoints (in background)
    - starts playback of the first song
    """
    global current_song_index
    with songs_lock:
        if not songs:
            return {"action": "SINGLE_PRESS", "error": "no songs"}
        current_song_index = 0
        song = songs[current_song_index]

    save_state()

    # Call power-on endpoints in background (do not block request)
    threading.Thread(target=_call_power_endpoints, args=("on",), daemon=True).start()

    # Start playback (best-effort)
    try:
        play_res = play_song(song)
    except Exception:
        logger.exception("Error starting playback on single_press")
        play_res = {"played": False, "error": "exception starting playback"}

    return {"action": "SINGLE_PRESS", "song": song, "play": play_res}


@app.post("/api/double_press")
def double_press():
    """DOUBLE_PRESS: call external endpoint to go to next song."""
    url = os.getenv("NEXT_SONG_URL", "http://192.168.1.183/next_song")
    # Advance to next song in the shuffled list
    global current_song_index
    with songs_lock:
        current_song_index = (current_song_index + 1) % len(songs)
        song = songs[current_song_index]

    # Persist the new index so restarts resume here.
    save_state()

    # Try to play the matching mp3 for this song (best-effort, non-blocking).
    try:
        play_res = play_song(song)
        logger.info("Playback result: %s", play_res)
    except Exception:
        # protect against any unexpected playback errors from bubbling up
        logger.exception("Error while trying to play song %s", song)
        play_res = {"played": False, "error": "playback exception"}

    logger.info("DOUBLE_PRESS triggered, next song: %s, forwarding to %s", song, url)
    res = _post_forward(url, json_payload={})
    logger.info(
        "DOUBLE_PRESS result: forwarded=%s status=%s error=%s",
        res.get("forwarded"),
        res.get("status"),
        res.get("error"),
    )
    # Return the song so the frontend can display it and include playback info
    return {"action": "DOUBLE_PRESS", "song": song, "play": play_res, **res}


@app.post("/api/multi_press")
def multi_press():
    """MULTI_PRESS: call external endpoint to show lights."""
    url = os.getenv("SHOW_LIGHTS_URL", "http://192.168.1.183/show_lights")
    res = _post_forward(url, json_payload={})
    return {"action": "MULTI_PRESS", **res}


@app.post("/api/long_press")
def long_press():
    """LONG_PRESS: stop playback and call power-off endpoints."""
    # Stop playback immediately
    _stop_playback()

    # Call power-off endpoints in background and return immediately
    threading.Thread(target=_call_power_endpoints, args=("off",), daemon=True).start()
    return {"action": "LONG_PRESS", "result": "power-off triggered"}


@app.get("/api/current_song")
def get_current_song():
    """Return the currently playing song and its index."""
    with songs_lock:
        if not songs:
            return {"song": None, "index": None}
        return {"song": songs[current_song_index], "index": current_song_index}


# Serve admin page at /admin (maps to static/admin.html)
@app.get("/admin")
def admin_page():
    admin_path = os.path.join(os.getcwd(), "static", "admin.html")
    if os.path.exists(admin_path):
        return FileResponse(admin_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="admin page not found")


# Serve the frontend from the `static` directory at the root path
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    # Allow running the app with: python app/main.py
    import uvicorn

    # Pass the app object directly so running `python app/main.py` works
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
