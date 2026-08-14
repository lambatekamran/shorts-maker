import base64
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Shorts Maker")

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# ---------------------------------------------------------------------------
# YouTube cookies (fixes "Sign in to confirm you're not a bot" on servers)
# ---------------------------------------------------------------------------
COOKIES_PATH = "/tmp/cookies.txt"

cookies_b64 = os.environ.get("YT_COOKIES_B64")
if cookies_b64:
    with open(COOKIES_PATH, "wb") as f:
        f.write(base64.b64decode(cookies_b64))


@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE_DIR / "static" / "index.html").read_text()


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def download_youtube(url: str, dest: Path) -> Path:
    out_template = str(dest / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "--merge-output-format", "mp4",
        "-o", out_template,
        url,
    ]
    if os.path.exists(COOKIES_PATH):
        cmd += ["--cookies", COOKIES_PATH]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    files = list(dest.glob("*.mp4"))
    if not files:
        raise RuntimeError("Download failed - no mp4 was produced")
    return files[0]


def get_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def extract_audio_energy(video_path: Path, window_sec: float = 1.0) -> np.ndarray:
    """Decode audio to mono 16kHz PCM and compute per-window RMS energy.

    RMS energy is used as a cheap, dependency-free proxy for 'excitement':
    louder / busier moments (cheering, shouting, music swells, impacts)
    score higher than quiet talk or silence.
    """
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    raw = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)

    sr = 16000
    window_size = int(sr * window_sec)
    if window_size == 0 or len(raw) == 0:
        return np.array([])

    n_windows = max(1, len(raw) // window_size)
    trimmed = raw[: n_windows * window_size]
    windows = trimmed.reshape(n_windows, window_size)
    return np.sqrt(np.mean(windows ** 2, axis=1) + 1e-9)


def pick_highlight_segments(rms, window_sec, clip_len, num_clips, total_duration):
    """Score every possible clip start by summed energy in that window,
    then greedily take the top N non-overlapping segments."""
    if len(rms) == 0:
        starts = np.linspace(0, max(0, total_duration - clip_len), num_clips)
        return [float(s) for s in starts]

    windows_per_clip = max(1, int(clip_len / window_sec))
    scores = []
    for start_idx in range(0, max(1, len(rms) - windows_per_clip + 1)):
        score = rms[start_idx:start_idx + windows_per_clip].sum()
        scores.append((score, start_idx))
    if not scores:
        return [0.0]

    scores.sort(key=lambda x: x[0], reverse=True)

    chosen = []
    used = np.zeros(len(rms), dtype=bool)
    for score, start_idx in scores:
        end_idx = start_idx + windows_per_clip
        if used[start_idx:end_idx].any():
            continue
        chosen.append(start_idx * window_sec)
        used[max(0, start_idx - windows_per_clip):end_idx] = True
        if len(chosen) >= num_clips:
            break

    chosen.sort()
    chosen = [min(c, max(0, total_duration - clip_len)) for c in chosen]
    return chosen


def make_vertical_clip(src: Path, start: float, duration: float, dest: Path):
    """Convert a segment into a 1080x1920 (9:16) short with a blurred
    background fill — the standard Reels/TikTok/Shorts look."""
    vf = (
        "split=2[bg][fg];"
        "[bg]scale=1080:1920,gblur=sigma=20[bg2];"
        "[fg]scale=1080:-2[fg2];"
        "[bg2][fg2]overlay=(W-w)/2:(H-h)/2,setsar=1"
    )
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-t", str(duration),
        "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        str(dest),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.post("/process")
async def process(
    youtube_url: str = Form(default=""),
    clip_length: int = Form(default=30),
    num_clips: int = Form(default=3),
    file: UploadFile = File(default=None),
):
    job_id = uuid.uuid4().hex[:8]
    work_dir = UPLOAD_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        if file is not None and file.filename:
            src_path = work_dir / file.filename
            with open(src_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        elif youtube_url.strip():
            try:
                src_path = download_youtube(youtube_url.strip(), work_dir)
            except subprocess.CalledProcessError as e:
                return {"error": f"Could not download that YouTube video: {e.stderr[-400:]}"}
        else:
            return {"error": "Provide a YouTube URL or upload a video file."}

        duration = get_duration(src_path)
        clip_length = max(1, min(clip_length, int(duration)))
        num_clips = max(1, min(num_clips, 10))

        rms = extract_audio_energy(src_path, window_sec=1.0)
        starts = pick_highlight_segments(rms, 1.0, clip_length, num_clips, duration)

        clips = []
        for i, start in enumerate(starts):
            out_name = f"{job_id}_short_{i + 1}.mp4"
            out_path = OUTPUT_DIR / out_name
            make_vertical_clip(src_path, start, clip_length, out_path)
            clips.append({
                "url": f"/outputs/{out_name}",
                "start": round(start, 1),
                "duration": clip_length,
            })

        return {"job_id": job_id, "clips": clips}

    except subprocess.CalledProcessError as e:
        return {"error": f"Processing failed: {e.stderr[-400:] if e.stderr else str(e)}"}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
