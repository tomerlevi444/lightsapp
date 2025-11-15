from fastapi import FastAPI, HTTPException, UploadFile, File
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
import time

# Configure basic logging so INFO/ERROR messages are visible on the console by default.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

app = FastAPI()

# Load configuration
def load_config():
    """Load configuration from config.json file."""
    config_path = os.path.join(os.getcwd(), "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.warning("Failed to load config.json, using defaults")
        return {}

config = load_config()

# Songs list - loaded from configured songs folder
songs = []

def load_songs_from_folder():
    """Load song list from configured songs folder."""
    main_folder = os.path.expanduser(config.get("main_folder", "~/Documents/midburn"))
    songs_dir = os.path.join(main_folder, "songs")
    if not os.path.exists(songs_dir):
        logger.warning("Songs directory not found: %s", songs_dir)
        return []
    
    # Supported audio file extensions
    audio_extensions = ['*.mp3', '*.wav', '*.m4a', '*.flac', '*.ogg']
    song_files = []
    
    for ext in audio_extensions:
        song_files.extend(glob.glob(os.path.join(songs_dir, ext)))
    
    # Extract just the filename without extension as song name
    song_names = [os.path.splitext(os.path.basename(f))[0] for f in song_files]
    logger.info("Found %d songs in %s", len(song_names), songs_dir)
    return song_names


def filter_songs_by_skip_count(all_songs: list[str]) -> list[str]:
    """Filter songs to include only those at or below the configured percentile by skip count.
    
    Songs not in skip_counts have a score of 0.
    Includes songs at percentile and below, including all songs with the same score.
    """
    if not all_songs:
        return []
    
    # Create list of (song, skip_count) tuples
    song_scores = [(song, skip_counts.get(song, 0)) for song in all_songs]
    
    # Sort by skip count (ascending)
    song_scores.sort(key=lambda x: x[1])
    
    # Get percentile from config (default 0.8 = 80th percentile)
    percentile = config.get("skip_percentile", 0.8)
    
    # Calculate percentile index
    percentile_index = int(len(song_scores) * percentile)
    if percentile_index >= len(song_scores):
        percentile_index = len(song_scores) - 1
    
    # Get the skip count at percentile (this is the inclusion threshold)
    threshold_score = song_scores[percentile_index][1]
    
    # Include all songs with skip count <= threshold
    filtered = [song for song, score in song_scores if score <= threshold_score]
    
    logger.info("Filtered %d/%d songs at %.0f%% percentile (including skip_count <= %d)", 
                len(filtered), len(all_songs), percentile * 100, threshold_score)
    return filtered
# index of the currently playing song in the songs list
current_song_index = 0
# flag to track if playback is active
playback_active = False
# timestamp when current song started playing
song_start_time: float | None = None
# skip counts for each song
skip_counts: dict[str, int] = {}
# lock to protect current_song_index and playback_active for concurrent access
songs_lock = threading.Lock()

# Playback process (non-blocking) and lock so we can stop previous playback
playback_process: subprocess.Popen | None = None
playback_lock = threading.Lock()
# generation counter to cancel watchers when playback is stopped
playback_generation = 0

# Clap playback processes
clap_processes: list[subprocess.Popen] = []
clap_lock = threading.Lock()

# Power control endpoints (will be updated dynamically)
POWER_URLS = []
POWER_OFF_URLS = []

def reload_power_urls():
    """Reload power URLs from current config."""
    global POWER_URLS, POWER_OFF_URLS
    power_host_prefix = config.get("power_host_prefix", "")
    power_hosts_str = os.getenv("POWER_HOSTS", config.get("power_hosts", ""))
    power_ports_str = config.get("power_ports", "/cm?cmnd=Power%20On,/cm?cmnd=Power%20Off")
    power_ports = power_ports_str.split(",")
    power_on_port = power_ports[0] if len(power_ports) > 0 else "/cm?cmnd=Power%20On"
    power_off_port = power_ports[1] if len(power_ports) > 1 else "/cm?cmnd=Power%20Off"
    
    if power_hosts_str:
        full_hosts = [f"{power_host_prefix}.{h}" if power_host_prefix else h for h in power_hosts_str.split(",")]
        POWER_URLS = [f"http://{h}{power_on_port}" for h in full_hosts]
        POWER_OFF_URLS = [f"http://{h}{power_off_port}" for h in full_hosts]
    else:
        POWER_URLS = []
        POWER_OFF_URLS = []
    logger.info("Power URLs reloaded: ON=%s OFF=%s", POWER_URLS, POWER_OFF_URLS)

reload_power_urls()

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
    """Persist current songs list, index, and skip counts to STATE_FILE atomically."""
    try:
        payload = {"songs": songs, "current_index": current_song_index, "skip_counts": skip_counts}
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
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
    """Find an audio file for the given song_name under configured songs folder and play it.

    song_name is the display name and files are expected to match the name.
    Returns the dict returned by _start_playback or an error dict.
    """
    main_folder = os.path.expanduser(config.get("main_folder", "~/Documents/midburn"))
    songs_dir = os.path.join(main_folder, "songs")
    
    # Try different audio extensions
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
    for ext in audio_extensions:
        file_path = os.path.join(songs_dir, f"{song_name}{ext}")
        if os.path.exists(file_path):
            return _start_playback(file_path)

    msg = f"song file not found for '{song_name}' in {songs_dir}"
    logger.info(msg)
    return {"played": False, "path": None, "error": msg}


def play_clap_sound():
    """Play clap sound without stopping current playback."""
    main_folder = os.path.expanduser(config.get("main_folder", "~/Documents/midburn"))
    clap_file = config.get("clap_file", "clap.wav")
    clap_path = os.path.join(main_folder, clap_file)
    if not os.path.exists(clap_path):
        logger.warning("Clap sound not found: %s", clap_path)
        return
    
    player = _which_player()
    if not player:
        logger.warning("No player available for clap sound")
        return
    
    if player == "afplay":
        args = [player, clap_path]
    elif player == "ffplay":
        args = [player, "-nodisp", "-autoexit", "-loglevel", "quiet", clap_path]
    elif player == "mpg123":
        args = [player, clap_path]
    elif player == "aplay":
        args = [player, clap_path]
    else:
        args = [player, clap_path]
    
    try:
        p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with clap_lock:
            clap_processes.append(p)
        logger.info("Playing clap sound")
    except Exception:
        logger.exception("Failed to play clap sound")


def _stop_clap_sounds():
    """Stop all clap sound processes."""
    with clap_lock:
        for p in clap_processes:
            try:
                if p.poll() is None:
                    p.terminate()
            except Exception:
                logger.exception("Failed to terminate clap process")
        clap_processes.clear()


def _call_power_endpoints(state: str) -> list:
    """Call the configured power endpoints with retry logic.

    state should be 'on' or 'off'. Returns list of results.
    Retries up to 2 times, logs only if all attempts fail.
    """
    results = []
    urls = POWER_URLS if state.lower() == "on" else POWER_OFF_URLS
    for u in urls:
        success = False
        last_error = None
        last_status = None
        
        for attempt in range(3):  # 3 attempts = initial + 2 retries
            try:
                resp = httpx.get(u, timeout=3.0)
                status = resp.status_code
                if 200 <= status < 300:
                    success = True
                    results.append({"url": u, "forwarded": True, "status": status, "error": None})
                    break
                else:
                    last_status = status
            except Exception as exc:
                last_error = str(exc)
        
        if not success:
            if last_status is not None:
                logger.error("Power endpoint %s failed after 3 attempts, last status: %s", u, last_status)
                results.append({"url": u, "forwarded": False, "status": last_status, "error": None})
            else:
                logger.error("Power endpoint %s failed after 3 attempts: %s", u, last_error)
                results.append({"url": u, "forwarded": False, "status": None, "error": last_error})
    return results


def _reset_system():
    """Reset system: stop playback, stop clap sounds, filter and shuffle songs, power off."""
    global playback_active, current_song_index
    
    # Stop playback and clap sounds
    _stop_playback()
    _stop_clap_sounds()
    
    with songs_lock:
        playback_active = False
        # Filter to lowest 80% by skip count and shuffle
        if songs:
            all_songs = songs.copy()
            filtered_songs = filter_songs_by_skip_count(all_songs)
            songs.clear()
            songs.extend(filtered_songs)
            random.shuffle(songs)
            current_song_index = 0
            logger.info("System reset: songs filtered and shuffled")
    
    # Power off (synchronously to ensure completion)
    _call_power_endpoints("off")


@app.on_event("startup")
def startup_event():
    """Set initial song at server startup."""
    global current_song_index, skip_counts
    
    # Load songs from folder
    folder_songs = load_songs_from_folder()
    
    # Load skip counts from persisted state
    state = load_state()
    if state and isinstance(state, dict):
        loaded_skip_counts = state.get("skip_counts", {})
        if isinstance(loaded_skip_counts, dict):
            skip_counts = loaded_skip_counts
    
    # Load all songs into list temporarily
    with songs_lock:
        songs.clear()
        songs.extend(folder_songs)
        if not songs:
            logger.warning("No songs found in ~/Documents/midburn/songs")
            return
    
    # Reset system (stop playback, filter, shuffle, power off)
    _reset_system()



@app.on_event("shutdown")
def shutdown_event():
    """Persist state on shutdown so restarts resume where we left off."""
    global playback_active
    
    # Stop playback and power off
    _stop_playback()
    with songs_lock:
        playback_active = False
    
    # Call power-off endpoints synchronously to ensure they complete before shutdown
    try:
        _call_power_endpoints("off")
        logger.info("Power-off endpoints called on shutdown")
    except Exception:
        logger.exception("Failed to call power-off endpoints on shutdown")
    
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
    - If playback is active: play clap sound along with current song
    - If playback is not active: start first song and call power-on endpoints
    """
    global current_song_index, playback_active, song_start_time
    
    with songs_lock:
        if not songs:
            return {"action": "SINGLE_PRESS", "error": "no songs"}
        
        # If playback is already active, just play clap sound
        if playback_active:
            play_clap_sound()
            return {"action": "SINGLE_PRESS", "clap": True, "song": songs[current_song_index]}
        
        # Otherwise start playback from first song
        current_song_index = 0
        song = songs[current_song_index]
        playback_active = True
        song_start_time = time.time()

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
    url = os.getenv("NEXT_SONG_URL", config.get("next_song_url", "http://192.168.1.183/next_song"))
    
    # Stop any clap sounds
    _stop_clap_sounds()
    
    # Advance to next song in the shuffled list
    global current_song_index, playback_active, song_start_time, skip_counts
    
    with songs_lock:
        if not songs:
            return {"action": "DOUBLE_PRESS", "error": "no songs available"}
        
        # Check if current song was skipped
        if playback_active and song_start_time is not None:
            elapsed = time.time() - song_start_time
            skip_threshold = config.get("skip_threshold_seconds", 20)
            if elapsed < skip_threshold:
                prev_song = songs[current_song_index]
                skip_counts[prev_song] = skip_counts.get(prev_song, 0) + 1
                logger.info("Song '%s' skipped after %.1f seconds (skip count: %d)", prev_song, elapsed, skip_counts[prev_song])
        
        current_song_index = (current_song_index + 1) % len(songs)
        song = songs[current_song_index]
        playback_active = True
        song_start_time = time.time()

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

    # logger.info("DOUBLE_PRESS triggered, next song: %s, forwarding to %s", song, url)
    # res = _post_forward(url, json_payload={})
    # logger.info(
    #     "DOUBLE_PRESS result: forwarded=%s status=%s error=%s",
    #     res.get("forwarded"),
    #     res.get("status"),
    #     res.get("error"),
    # )
    # Return the song so the frontend can display it and include playback info
    return {"action": "DOUBLE_PRESS", "song": song, "play": play_res}


@app.post("/api/multi_press")
def multi_press():
    """MULTI_PRESS: call external endpoint to show lights."""
    url = os.getenv("SHOW_LIGHTS_URL", config.get("show_lights_url", "http://192.168.1.183/show_lights"))
    res = _post_forward(url, json_payload={})
    return {"action": "MULTI_PRESS", **res}


@app.post("/api/long_press")
def long_press():
    """LONG_PRESS: stop playback, shuffle songs, and call power-off endpoints."""
    # Reset system (stop playback, shuffle, power off)
    _reset_system()
    save_state()
    return {"action": "LONG_PRESS", "result": "power-off triggered"}


@app.get("/api/current_song")
def get_current_song():
    """Return the currently playing song and its index."""
    with songs_lock:
        if not songs or not playback_active:
            return {"song": None, "index": None}
        return {"song": songs[current_song_index], "index": current_song_index}


@app.post("/api/clear_skip_counts")
def clear_skip_counts():
    """Clear all skip counts and recalculate/shuffle songs."""
    global skip_counts, current_song_index
    
    # Clear skip counts
    skip_counts = {}
    
    # Reload, filter and shuffle songs
    folder_songs = load_songs_from_folder()
    with songs_lock:
        filtered_songs = filter_songs_by_skip_count(folder_songs)
        songs.clear()
        songs.extend(filtered_songs)
        random.shuffle(songs)
        current_song_index = 0
    
    save_state()
    logger.info("Skip counts cleared, songs recalculated and shuffled")
    return {"action": "CLEAR_SKIP_COUNTS", "result": "success"}


@app.get("/api/stats")
def get_stats():
    """Return statistics including skip counts, total songs, and excluded songs."""
    # Get total number of songs from folder
    folder_songs = load_songs_from_folder()
    
    # Get filtered songs (those that will be played)
    filtered_songs = filter_songs_by_skip_count(folder_songs)
    
    # Find excluded songs
    excluded_songs = [song for song in folder_songs if song not in filtered_songs]
    excluded_with_counts = [(song, skip_counts.get(song, 0)) for song in excluded_songs]
    excluded_with_counts.sort(key=lambda x: x[1], reverse=True)
    
    with songs_lock:
        return {
            "skip_counts": skip_counts.copy(),
            "total_songs": len(folder_songs),
            "excluded_songs": excluded_with_counts
        }


@app.get("/api/config")
def get_config():
    """Return the current configuration."""
    return config


def save_config():
    """Save configuration to config.json file and reload it."""
    global config
    config_path = os.path.join(os.getcwd(), "config.json")
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        config = load_config()
        logger.info("Configuration saved and reloaded")
    except Exception:
        logger.exception("Failed to save config.json")


@app.post("/api/add_power_host")
async def add_power_host(request: dict):
    """Add a power host to the configuration."""
    host = request.get("host", "").strip()
    if not host:
        return {"result": "error", "message": "Host number required"}
    
    # Get current hosts
    current_hosts = config.get("power_hosts", "")
    hosts_list = [h.strip() for h in current_hosts.split(",") if h.strip()]
    
    # Add new host if not already present
    if host not in hosts_list:
        hosts_list.append(host)
        config["power_hosts"] = ",".join(hosts_list)
        save_config()
        reload_power_urls()
        logger.info("Added power host: %s", host)
        return {"result": "success", "host": host}
    else:
        return {"result": "error", "message": "Host already exists"}


@app.post("/api/remove_power_host")
async def remove_power_host(request: dict):
    """Remove a power host from the configuration."""
    host = request.get("host", "").strip()
    if not host:
        return {"result": "error", "message": "Host number required"}
    
    # Get current hosts
    current_hosts = config.get("power_hosts", "")
    hosts_list = [h.strip() for h in current_hosts.split(",") if h.strip()]
    
    # Remove host if present
    if host in hosts_list:
        hosts_list.remove(host)
        config["power_hosts"] = ",".join(hosts_list)
        save_config()
        reload_power_urls()
        logger.info("Removed power host: %s", host)
        return {"result": "success", "host": host}
    else:
        return {"result": "error", "message": "Host not found"}


@app.post("/api/upload_songs")
async def upload_songs(files: list[UploadFile] = File(...)):
    """Upload MP3 files to songs folder."""
    global current_song_index
    main_folder = os.path.expanduser(config.get("main_folder", "~/Documents/midburn"))
    songs_dir = os.path.join(main_folder, "songs")
    
    # Ensure songs directory exists
    os.makedirs(songs_dir, exist_ok=True)
    
    uploaded_count = 0
    for file in files:
        if not file.filename.endswith('.mp3'):
            logger.warning("Skipping non-MP3 file: %s", file.filename)
            continue
        
        try:
            file_path = os.path.join(songs_dir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            uploaded_count += 1
            logger.info("Uploaded song: %s", file.filename)
        except Exception:
            logger.exception("Failed to upload file: %s", file.filename)
    
    # Reload, filter and shuffle songs
    folder_songs = load_songs_from_folder()
    with songs_lock:
        filtered_songs = filter_songs_by_skip_count(folder_songs)
        songs.clear()
        songs.extend(filtered_songs)
        random.shuffle(songs)
        current_song_index = 0
    save_state()
    logger.info("Songs recalculated and shuffled after upload")
    
    return {"action": "UPLOAD_SONGS", "uploaded": uploaded_count}


# Serve admin page at /admin (maps to static/admin.html)
@app.get("/admin")
def admin_page():
    admin_path = os.path.join(os.getcwd(), "static", "admin.html")
    if os.path.exists(admin_path):
        return FileResponse(admin_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="admin page not found")


# Serve stats page at /stats (maps to static/stats.html)
@app.get("/stats")
def stats_page():
    stats_path = os.path.join(os.getcwd(), "static", "stats.html")
    if os.path.exists(stats_path):
        return FileResponse(stats_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="stats page not found")


# Serve config page at /config (maps to static/config.html)
@app.get("/config")
def config_page():
    config_path = os.path.join(os.getcwd(), "static", "config.html")
    if os.path.exists(config_path):
        return FileResponse(config_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="config page not found")


# Serve the frontend from the `static` directory at the root path
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    # Allow running the app with: python app/main.py
    import uvicorn

    # Pass the app object directly so running `python app/main.py` works
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
