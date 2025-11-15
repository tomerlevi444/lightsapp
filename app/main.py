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
import socket

# Configure basic logging so INFO/ERROR messages are visible on the console by default.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Create a string buffer to capture logs
log_buffer = []
max_log_lines = 1000

class LogHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        log_buffer.append(log_entry)
        if len(log_buffer) > max_log_lines:
            log_buffer.pop(0)

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Add custom handler to capture logs
log_handler = LogHandler()
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(log_handler)

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

def get_sdcard_path():
    """Find SD card mount path on Raspberry Pi."""
    if sys.platform.startswith("linux"):
        # Check common Raspberry Pi SD card mount points
        sdcard_paths = ["/mnt/sdcard", "/media/sdcard"]
        for path in sdcard_paths:
            if os.path.exists(path) and os.path.ismount(path):
                return path
        # Check /media for any mounted device
        media_dir = "/media"
        if os.path.exists(media_dir):
            for user_dir in os.listdir(media_dir):
                user_path = os.path.join(media_dir, user_dir)
                if os.path.isdir(user_path):
                    devices = os.listdir(user_path)
                    if devices:
                        return os.path.join(user_path, devices[0])
    return None

def get_main_folder():
    """Get main folder path, using SD card if configured."""
    if config.get("use_sdcard", False):
        sdcard_path = get_sdcard_path()
        if sdcard_path:
            logger.info("Using SD card: %s", sdcard_path)
            return sdcard_path
        else:
            logger.warning("SD card mode enabled but no SD card found, falling back to local folder")
    return os.path.expanduser(config.get("main_folder", "~/Documents/midburn"))

def get_local_ip():
    """Get local IP address in LAN."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

config = load_config()

# Songs list - loaded from configured songs folder
songs = []
# Claps lists - loaded from claps subfolders
start_claps = []
middle_claps = []
end_claps = []
current_start_clap_index = 0
current_middle_clap_index = 0
current_end_clap_index = 0

def load_songs_from_folder():
    """Load song list from configured songs folder."""
    main_folder = get_main_folder()
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


def load_claps_from_subfolder(subfolder_name: str):
    """Load clap list from a specific subfolder and shuffle."""
    main_folder = get_main_folder()
    claps_folder = config.get("claps_folder", "claps")
    claps_dir = os.path.join(main_folder, claps_folder, subfolder_name)
    if not os.path.exists(claps_dir):
        logger.warning("Claps subfolder not found: %s", claps_dir)
        return []
    
    # Supported audio file extensions
    audio_extensions = ['*.mp3', '*.wav', '*.m4a', '*.flac', '*.ogg']
    clap_files = []
    
    for ext in audio_extensions:
        clap_files.extend(glob.glob(os.path.join(claps_dir, ext)))
    
    # Extract just the filename without extension as clap name
    clap_names = [os.path.splitext(os.path.basename(f))[0] for f in clap_files]
    random.shuffle(clap_names)
    logger.info("Found %d claps in %s/%s", len(clap_names), claps_folder, subfolder_name)
    return clap_names


def filter_songs_by_skip_count(all_songs: list[str]) -> list[str]:
    """Filter songs to include only those at or below the configured percentile by skip count.
    
    Songs not in skip_counts have a score of 0.
    Includes songs at percentile and below, including all songs with the same score.
    If filter_unpopular_songs is disabled, returns all songs.
    """
    if not all_songs:
        return []
    
    # If filtering is disabled, return all songs
    if not config.get("filter_unpopular_songs", True):
        logger.info("Song filtering disabled, using all %d songs", len(all_songs))
        return all_songs
    
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
# flag to skip claps on next song transition
skip_next_claps = False
# lock to protect current_song_index and playback_active for concurrent access
songs_lock = threading.Lock()

# Playback process (non-blocking) and lock so we can stop previous playback
playback_process: subprocess.Popen | None = None
playback_lock = threading.Lock()
# generation counter to cancel watchers when playback is stopped
playback_generation = 0

# Multi-press timer management
multi_press_timer = None
multi_press_lock = threading.Lock()

# Clap playback processes
clap_processes: list[subprocess.Popen] = []
clap_lock = threading.Lock()

# Power control endpoints (will be updated dynamically)
POWER_URLS = []
POWER_OFF_URLS = []

def reload_power_urls():
    """Reload power URLs from current config."""
    global POWER_URLS, POWER_OFF_URLS
    # Get prefix from local IP (e.g., "192.168.1.182" -> "192.168.1")
    local_ip = config.get("local_ip", "127.0.0.1")
    power_host_prefix = ".".join(local_ip.split(".")[:3])
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
PLAYLISTS_FILE = os.path.join(os.getcwd(), "playlists.json")


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
    """Persist current songs list, index, skip counts, claps, and clap indices to STATE_FILE atomically."""
    try:
        payload = {
            "songs": songs,
            "current_index": current_song_index,
            "skip_counts": skip_counts,
            "start_claps": start_claps,
            "middle_claps": middle_claps,
            "end_claps": end_claps,
            "current_start_clap_index": current_start_clap_index,
            "current_middle_clap_index": current_middle_clap_index,
            "current_end_clap_index": current_end_clap_index
        }
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, STATE_FILE)
    except Exception:
        logger.exception("Failed to save state to %s", STATE_FILE)


def load_playlists() -> dict:
    """Load playlists configuration from PLAYLISTS_FILE."""
    if not os.path.exists(PLAYLISTS_FILE):
        return {}
    try:
        with open(PLAYLISTS_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        logger.exception("Failed to load playlists from %s", PLAYLISTS_FILE)
        return {}


def save_playlists(playlists_data: dict) -> None:
    """Save playlists configuration to PLAYLISTS_FILE."""
    try:
        tmp = PLAYLISTS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(playlists_data, fh, indent=2)
        os.replace(tmp, PLAYLISTS_FILE)
    except Exception:
        logger.exception("Failed to save playlists to %s", PLAYLISTS_FILE)


def _which_player() -> Optional[str]:
    """Return the first available playback command on the system or None.

    Preference: macOS 'afplay', then 'ffplay', then 'mpg123'. On Linux prefer
    'mpg123' or 'ffplay'."""
    candidates = []
    if sys.platform == "darwin":
        candidates = ["afplay", "ffplay", "mpg123"]
    elif sys.platform.startswith("linux"):
        candidates = ["mpg123", "ffplay", "omxplayer", "aplay"]
    else:
        candidates = ["ffplay", "mpg123", "afplay"]

    for cmd in candidates:
        if shutil.which(cmd):
            logger.info("Found audio player: %s", cmd)
            return cmd
    logger.error("No audio player found. Tried: %s", candidates)
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

    # Build args depending on player with 80% volume
    if player == "afplay":
        args = [player, "-v", "0.8", file_path]
    elif player == "ffplay":
        args = [player, "-nodisp", "-autoexit", "-loglevel", "quiet", "-volume", "80", file_path]
    elif player == "mpg123":
        args = [player, "-f", "26214", file_path]  # 80% of 32768
    elif player == "omxplayer":
        args = [player, "--vol", "-600", file_path]  # Raspberry Pi player
    elif player == "aplay":
        args = [player, file_path]
    else:
        args = [player, file_path]
    
    logger.info("Starting playback with command: %s", " ".join(args))

    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error("Audio file not found: %s", file_path)
            return {"played": False, "path": file_path, "error": "file not found"}
        
        # Stop previous and start new playback
        _stop_playback()
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
                # Wait for process to finish
                proc.wait()
                
                # Process has exited, check for errors
                stdout, stderr = proc.communicate()
                if proc.returncode != 0:
                    logger.error("Playback process failed with code %d. stderr: %s", proc.returncode, stderr.decode() if stderr else "none")
            except Exception:
                logger.exception("Watcher error waiting for playback process")
            finally:
                with playback_lock:
                    current_gen = playback_generation
                logger.info("Playback process exited (gen=%s current_gen=%s)", gen, current_gen)
                if gen == current_gen:
                    # Play end clap only if not skipped
                    global playback_active, current_song_index, skip_next_claps
                    if not skip_next_claps:
                        play_end_clap()
                    # Mark playback as inactive and advance to next song
                    with songs_lock:
                        playback_active = False
                        if songs:
                            current_song_index = (current_song_index + 1) % len(songs)
                    save_state()
                    logger.info("Playback finished; marked as inactive and advanced to next song")
                    # Only call power endpoints if this playback wasn't cancelled
                    logger.info("Playback finished; calling power-off endpoints")
                    # call in background so watcher returns quickly
                    threading.Thread(target=_call_power_endpoints, args=("off",), daemon=True).start()

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
    main_folder = get_main_folder()
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


def play_clap_from_folder(clap_list: list, index_var_name: str, subfolder: str):
    """Play next clap sound from specified clap list and subfolder."""
    if not clap_list:
        logger.warning("No claps available in %s", subfolder)
        return
    
    # Get current index from globals
    if index_var_name == "start":
        global current_start_clap_index
        current_index = current_start_clap_index
    elif index_var_name == "middle":
        global current_middle_clap_index
        current_index = current_middle_clap_index
    else:  # end
        global current_end_clap_index
        current_index = current_end_clap_index
    
    main_folder = get_main_folder()
    claps_folder = config.get("claps_folder", "claps")
    claps_dir = os.path.join(main_folder, claps_folder, subfolder)
    
    # Get current clap and advance index
    clap_name = clap_list[current_index]
    new_index = (current_index + 1) % len(clap_list)
    
    # Update the appropriate global index
    if index_var_name == "start":
        current_start_clap_index = new_index
    elif index_var_name == "middle":
        current_middle_clap_index = new_index
    else:
        current_end_clap_index = new_index
    
    save_state()
    
    # Try different audio extensions
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
    clap_path = None
    for ext in audio_extensions:
        test_path = os.path.join(claps_dir, f"{clap_name}{ext}")
        if os.path.exists(test_path):
            clap_path = test_path
            break
    
    if not clap_path:
        logger.warning("Clap sound not found: %s in %s", clap_name, subfolder)
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
        logger.info("Playing %s clap sound: %s", subfolder, clap_name)
    except Exception:
        logger.exception("Failed to play clap sound")


def play_start_clap():
    """Play a clap from the start folder."""
    play_clap_from_folder(start_claps, "start", "start")


def play_middle_clap():
    """Play a clap from the middle folder."""
    play_clap_from_folder(middle_claps, "middle", "middle")


def play_end_clap():
    """Play a clap from the end folder."""
    play_clap_from_folder(end_claps, "end", "end")


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
            # Ensure current_song_index is valid after shuffle
            if current_song_index >= len(songs):
                current_song_index = 0
            logger.info("System reset: songs filtered and shuffled")
    
    # Power off (synchronously to ensure completion)
    _call_power_endpoints("off")


@app.on_event("startup")
def startup_event():
    """Set initial song at server startup."""
    global current_song_index, skip_counts, current_clap_index
    
    # Detect and save local IP address
    local_ip = get_local_ip()
    config["local_ip"] = local_ip
    save_config()
    logger.info("Local IP address: %s", local_ip)
    
    # Load songs from folder
    folder_songs = load_songs_from_folder()
    
    # Load claps from subfolders and shuffle
    start_claps.clear()
    start_claps.extend(load_claps_from_subfolder("start"))
    middle_claps.clear()
    middle_claps.extend(load_claps_from_subfolder("middle"))
    end_claps.clear()
    end_claps.extend(load_claps_from_subfolder("end"))
    
    # Load skip counts and clap indices from persisted state
    state = load_state()
    if state and isinstance(state, dict):
        loaded_skip_counts = state.get("skip_counts", {})
        if isinstance(loaded_skip_counts, dict):
            skip_counts = loaded_skip_counts
        
        loaded_start_index = state.get("current_start_clap_index", 0)
        if isinstance(loaded_start_index, int):
            current_start_clap_index = loaded_start_index
        
        loaded_middle_index = state.get("current_middle_clap_index", 0)
        if isinstance(loaded_middle_index, int):
            current_middle_clap_index = loaded_middle_index
        
        loaded_end_index = state.get("current_end_clap_index", 0)
        if isinstance(loaded_end_index, int):
            current_end_clap_index = loaded_end_index
    
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


@app.get("/api/single_press")
def single_press():
    """SINGLE_PRESS: play start clap, then current song, and call power-on endpoints, only if not already playing."""
    global current_song_index, playback_active, song_start_time
    
    with songs_lock:
        if not songs:
            return {"action": "SINGLE_PRESS", "error": "no songs"}
        
        # Do nothing if already playing
        if playback_active:
            return {"action": "SINGLE_PRESS", "error": "already playing"}
        
        # Start playback from current index
        song = songs[current_song_index]
        playback_active = True
        song_start_time = time.time()

    save_state()

    # Call power-on endpoints in background (do not block request)
    threading.Thread(target=_call_power_endpoints, args=("on",), daemon=True).start()
    
    # Cancel any pending multi-press timer
    _cancel_multi_press_timer()
    
    # Call stage ESP32 endpoint
    threading.Thread(target=lambda: _post_forward("http://stage-esp32.local/show"), daemon=True).start()

    # Play start clap and song simultaneously
    play_start_clap()
    try:
        play_song(song)
    except Exception:
        logger.exception("Error starting playback with start clap")

    return {"action": "SINGLE_PRESS", "song": song, "play": {"played": True}}


@app.get("/api/double_press")
def double_press():
    """DOUBLE_PRESS: go to next song only if playback is active, skip claps."""
    # Stop any clap sounds
    _stop_clap_sounds()
    
    # Advance to next song in the shuffled list
    global current_song_index, playback_active, song_start_time, skip_counts, skip_next_claps
    
    with songs_lock:
        if not songs:
            return {"action": "DOUBLE_PRESS", "error": "no songs available"}
        
        # Only advance if playback is active
        if not playback_active:
            return {"action": "DOUBLE_PRESS", "error": "no song playing"}
        
        # Check if current song was skipped
        if song_start_time is not None:
            elapsed = time.time() - song_start_time
            skip_threshold = config.get("skip_threshold_seconds", 20)
            if elapsed < skip_threshold:
                prev_song = songs[current_song_index]
                skip_counts[prev_song] = skip_counts.get(prev_song, 0) + 1
                logger.info("Song '%s' skipped after %.1f seconds (skip count: %d)", prev_song, elapsed, skip_counts[prev_song])
        
        current_song_index = (current_song_index + 1) % len(songs)
        song = songs[current_song_index]
        song_start_time = time.time()

    # Set flag to skip claps on this transition
    skip_next_claps = True
    
    # Persist the new index so restarts resume here.
    save_state()

    # Play song directly without start clap
    try:
        play_song(song)
        logger.info("Playback result for: %s", song)
    except Exception:
        logger.exception("Error while trying to play song %s", song)
    
    # Reset flag after starting playback
    skip_next_claps = False
    
    # Cancel any pending multi-press timer
    _cancel_multi_press_timer()
    
    # Call stage ESP32 endpoint only after successful action
    threading.Thread(target=lambda: _post_forward("http://stage-esp32.local/skip"), daemon=True).start()

    # Return the song so the frontend can display it and include playback info
    return {"action": "DOUBLE_PRESS", "song": song, "play": {"played": True}}


def _cancel_multi_press_timer():
    """Cancel pending multi-press timer."""
    global multi_press_timer
    with multi_press_lock:
        if multi_press_timer:
            multi_press_timer.cancel()
            multi_press_timer = None

@app.get("/api/multi_press")
def multi_press():
    """MULTI_PRESS: play next clap sound from middle folder only if song is playing."""
    global multi_press_timer
    
    # Cancel any existing timer
    _cancel_multi_press_timer()
    
    with songs_lock:
        if not playback_active:
            return {"action": "MULTI_PRESS", "error": "no song playing"}
        
        current_song = songs[current_song_index]
    
    play_middle_clap()
    
    # Call stage ESP32 endpoint only when action triggers
    threading.Thread(target=lambda: _post_forward("http://stage-esp32.local/special"), daemon=True).start()
    
    # Set timer for second call after 10 seconds
    with multi_press_lock:
        multi_press_timer = threading.Timer(10.0, lambda: _post_forward("http://stage-esp32.local/show"))
        multi_press_timer.start()
    
    return {"action": "MULTI_PRESS", "clap": True, "song": current_song}


@app.get("/api/long_press")
def long_press():
    """LONG_PRESS: stop playback, shuffle songs, and call power-off endpoints."""
    with songs_lock:
        # Do nothing if already in Ready state
        if not playback_active:
            return {"action": "LONG_PRESS", "result": "already stopped"}
    
    # Cancel any pending multi-press timer
    _cancel_multi_press_timer()
    
    # Call stage ESP32 endpoint
    threading.Thread(target=lambda: _post_forward("http://stage-esp32.local/idle"), daemon=True).start()
    
    # Reset system (stop playback, shuffle, power off)
    _reset_system()
    save_state()
    return {"action": "LONG_PRESS", "result": "power-off triggered"}


@app.get("/api/current_song")
def get_current_song():
    """Return the currently playing song and its index, or next song if not playing."""
    with songs_lock:
        if not songs:
            return {"song": None, "index": None, "next_song": None}
        if not playback_active:
            return {"song": None, "index": None, "next_song": songs[current_song_index]}
        return {"song": songs[current_song_index], "index": current_song_index, "next_song": None}


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


@app.post("/api/toggle_filter")
async def toggle_filter():
    """Toggle filter_unpopular_songs setting and reshuffle songs."""
    global current_song_index
    
    # Toggle the setting
    current_value = config.get("filter_unpopular_songs", True)
    config["filter_unpopular_songs"] = not current_value
    save_config()
    
    # Reload, filter and shuffle songs
    folder_songs = load_songs_from_folder()
    with songs_lock:
        filtered_songs = filter_songs_by_skip_count(folder_songs)
        songs.clear()
        songs.extend(filtered_songs)
        random.shuffle(songs)
        current_song_index = 0
    save_state()
    
    new_value = config.get("filter_unpopular_songs")
    logger.info("Filter unpopular songs: %s", new_value)
    return {"action": "TOGGLE_FILTER", "enabled": new_value}


@app.post("/api/upload_songs")
async def upload_songs(files: list[UploadFile] = File(...)):
    """Upload MP3 files to songs folder."""
    global current_song_index
    main_folder = get_main_folder()
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


@app.post("/api/create_playlist")
async def create_playlist(request: dict):
    """Create a new playlist folder."""
    playlist_name = request.get("name", "").strip()
    if not playlist_name:
        return {"result": "error", "message": "Playlist name required"}
    
    main_folder = get_main_folder()
    playlists_folder = config.get("playlists_folder", "playlists")
    playlist_path = os.path.join(main_folder, playlists_folder, playlist_name)
    
    if os.path.exists(playlist_path):
        return {"result": "error", "message": "Playlist already exists"}
    
    try:
        os.makedirs(playlist_path, exist_ok=True)
        logger.info("Created playlist: %s", playlist_name)
        return {"result": "success", "name": playlist_name}
    except Exception:
        logger.exception("Failed to create playlist: %s", playlist_name)
        return {"result": "error", "message": "Failed to create playlist"}


@app.get("/api/playlists")
def get_playlists():
    """Get list of all playlists."""
    main_folder = get_main_folder()
    playlists_folder = config.get("playlists_folder", "playlists")
    playlists_dir = os.path.join(main_folder, playlists_folder)
    
    if not os.path.exists(playlists_dir):
        return {"playlists": []}
    
    playlists = []
    for item in os.listdir(playlists_dir):
        item_path = os.path.join(playlists_dir, item)
        if os.path.isdir(item_path):
            playlists.append(item)
    
    return {"playlists": sorted(playlists)}


@app.get("/api/playlist/{playlist_name}")
def get_playlist_songs(playlist_name: str):
    """Get songs in a playlist in order."""
    main_folder = get_main_folder()
    playlists_folder = config.get("playlists_folder", "playlists")
    playlist_path = os.path.join(main_folder, playlists_folder, playlist_name)
    
    if not os.path.exists(playlist_path):
        raise HTTPException(status_code=404, detail="Playlist not found")
    
    # Load playlist order from playlists.json
    playlists_data = load_playlists()
    ordered_songs = playlists_data.get(playlist_name, [])
    
    # Get all songs from folder
    audio_extensions = ['*.mp3', '*.wav', '*.m4a', '*.flac', '*.ogg']
    song_files = []
    for ext in audio_extensions:
        song_files.extend(glob.glob(os.path.join(playlist_path, ext)))
    
    all_songs = set(os.path.splitext(os.path.basename(f))[0] for f in song_files)
    
    # Add any new songs not in order list
    for song in all_songs:
        if song not in ordered_songs:
            ordered_songs.append(song)
    
    # Remove songs that no longer exist
    ordered_songs = [s for s in ordered_songs if s in all_songs]
    
    # Save updated order
    playlists_data[playlist_name] = ordered_songs
    save_playlists(playlists_data)
    
    return {"playlist": playlist_name, "songs": ordered_songs}


@app.post("/api/playlist/{playlist_name}/upload")
async def upload_to_playlist(playlist_name: str, files: list[UploadFile] = File(...)):
    """Upload songs to a playlist."""
    main_folder = get_main_folder()
    playlists_folder = config.get("playlists_folder", "playlists")
    playlist_path = os.path.join(main_folder, playlists_folder, playlist_name)
    
    if not os.path.exists(playlist_path):
        raise HTTPException(status_code=404, detail="Playlist not found")
    
    uploaded_count = 0
    new_songs = []
    for file in files:
        if not file.filename.endswith('.mp3'):
            logger.warning("Skipping non-MP3 file: %s", file.filename)
            continue
        
        try:
            file_path = os.path.join(playlist_path, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            uploaded_count += 1
            new_songs.append(os.path.splitext(file.filename)[0])
            logger.info("Uploaded song to playlist %s: %s", playlist_name, file.filename)
        except Exception:
            logger.exception("Failed to upload file: %s", file.filename)
    
    # Add new songs to end of playlist order
    if new_songs:
        playlists_data = load_playlists()
        ordered_songs = playlists_data.get(playlist_name, [])
        for song in new_songs:
            if song not in ordered_songs:
                ordered_songs.append(song)
        playlists_data[playlist_name] = ordered_songs
        save_playlists(playlists_data)
    
    return {"action": "UPLOAD_TO_PLAYLIST", "uploaded": uploaded_count}


@app.post("/api/playlist/{playlist_name}/move")
async def move_song_in_playlist(playlist_name: str, request: dict):
    """Move a song up or down in playlist order."""
    song = request.get("song", "")
    direction = request.get("direction", "")
    
    if not song or direction not in ["up", "down"]:
        return {"result": "error", "message": "Invalid parameters"}
    
    playlists_data = load_playlists()
    ordered_songs = playlists_data.get(playlist_name, [])
    
    if song not in ordered_songs:
        return {"result": "error", "message": "Song not found"}
    
    idx = ordered_songs.index(song)
    
    if direction == "up" and idx > 0:
        ordered_songs[idx], ordered_songs[idx - 1] = ordered_songs[idx - 1], ordered_songs[idx]
    elif direction == "down" and idx < len(ordered_songs) - 1:
        ordered_songs[idx], ordered_songs[idx + 1] = ordered_songs[idx + 1], ordered_songs[idx]
    else:
        return {"result": "error", "message": "Cannot move in that direction"}
    
    playlists_data[playlist_name] = ordered_songs
    save_playlists(playlists_data)
    
    return {"result": "success", "songs": ordered_songs}


@app.post("/api/playlist/{playlist_name}/play")
async def play_playlist(playlist_name: str):
    """Load and start playing a playlist."""
    global current_song_index, playback_active
    
    # Stop current playback
    _stop_playback()
    _stop_clap_sounds()
    
    # Load playlist songs
    playlists_data = load_playlists()
    ordered_songs = playlists_data.get(playlist_name, [])
    
    if not ordered_songs:
        return {"result": "error", "message": "Playlist is empty"}
    
    # Replace current songs with playlist songs
    with songs_lock:
        songs.clear()
        songs.extend(ordered_songs)
        current_song_index = 0
        playback_active = False
    
    save_state()
    logger.info("Loaded playlist: %s with %d songs", playlist_name, len(ordered_songs))
    
    return {"result": "success", "playlist": playlist_name, "songs": len(ordered_songs)}


@app.post("/api/playlist/{playlist_name}/delete")
async def delete_playlist(playlist_name: str):
    """Delete a playlist folder and its contents."""
    main_folder = get_main_folder()
    playlists_folder = config.get("playlists_folder", "playlists")
    playlist_path = os.path.join(main_folder, playlists_folder, playlist_name)
    
    if not os.path.exists(playlist_path):
        return {"result": "error", "message": "Playlist not found"}
    
    try:
        shutil.rmtree(playlist_path)
        logger.info("Deleted playlist: %s", playlist_name)
        
        # Remove from playlists.json
        playlists_data = load_playlists()
        if playlist_name in playlists_data:
            del playlists_data[playlist_name]
            save_playlists(playlists_data)
        
        return {"result": "success", "playlist": playlist_name}
    except Exception:
        logger.exception("Failed to delete playlist: %s", playlist_name)
        return {"result": "error", "message": "Failed to delete playlist"}


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


# Serve playlist page at /playlist (maps to static/playlist.html)
@app.get("/playlist")
def playlist_page():
    playlist_path = os.path.join(os.getcwd(), "static", "playlist.html")
    if os.path.exists(playlist_path):
        return FileResponse(playlist_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="playlist page not found")


# Serve logs page at /logs (maps to static/logs.html)
@app.get("/logs")
def logs_page():
    logs_path = os.path.join(os.getcwd(), "static", "logs.html")
    if os.path.exists(logs_path):
        return FileResponse(logs_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="logs page not found")


@app.get("/api/logs")
def get_logs():
    """Get recent log entries from the application."""
    try:
        return {"logs": "\n".join(log_buffer[-100:])}
    except Exception as e:
        return {"logs": f"Error reading logs: {str(e)}"}





# Serve the frontend from the `static` directory at the root path
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    # Allow running the app with: python app/main.py
    import uvicorn

    # Pass the app object directly so running `python app/main.py` works
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
