---
title: Shorts Maker
emoji: 🎬
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Shorts Maker

Turn a YouTube link or an uploaded video into vertical (9:16) shorts,
automatically picked from the most "exciting" (loudest / most energetic)
moments in the audio track.

## How it works

1. **Input**: a YouTube URL (downloaded with `yt-dlp`) or an uploaded video file.
2. **Highlight detection**: the audio track is decoded and split into
   1-second windows. Each window's RMS (loudness) energy is computed with
   `numpy`. This is a simple, dependency-free heuristic — cheering, shouting,
   music swells, and impacts score higher than quiet talk or silence — not a
   real AI/ML model. The top N non-overlapping windows become your clips.
3. **Vertical conversion**: each clip is rendered to 1080x1920 with `ffmpeg`,
   using a blurred, scaled copy of the frame as a background fill (the
   standard Reels/TikTok/Shorts look) so nothing gets stretched or cropped away.

## Run it locally

```bash
pip install -r requirements.txt
# ffmpeg must be installed and on PATH (brew install ffmpeg / apt install ffmpeg)
uvicorn app:app --reload
# open http://localhost:8000
```

## Deploy for free — Hugging Face Spaces (recommended)

Hugging Face Spaces gives you a free CPU tier (2 vCPU, 16 GB RAM) with Docker
support, so `ffmpeg` and `yt-dlp` work out of the box. No credit card, no
trial expiry. The free tier sleeps after ~48h of inactivity and wakes up
automatically on the next visit.

1. Create a free account at https://huggingface.co/join
2. Click **New Space** → choose **Docker** as the SDK → pick the free
   **CPU basic** hardware.
3. Push this folder's contents to the Space's git repo:
   ```bash
   git init
   git remote add origin https://huggingface.co/spaces/<your-username>/shorts-maker
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```
4. The Space will build the Dockerfile automatically and give you a public
   URL like `https://<your-username>-shorts-maker.hf.space`.

## Deploy alternative — Render (free tier)

Render also has a free Docker web service tier. It sleeps after ~15 minutes
of inactivity (slower cold starts than Hugging Face) and has less free CPU,
but works the same way:

1. Push this folder to a GitHub repo.
2. On https://render.com, choose **New Web Service** → connect the repo →
   Render will detect the Dockerfile automatically.
3. Set the instance type to **Free**.

## Notes & limits

- Output videos are written to `outputs/` and served statically — on free
  hosting tiers this storage is **not persistent**, so treat downloads as
  ephemeral (the app is meant for on-demand generation, not as permanent storage).
- Downloading YouTube videos via `yt-dlp` sits in a legal gray area relative
  to YouTube's Terms of Service, even though the tool is widely used for
  personal/fair-use clipping. Worth keeping in mind if this becomes a
  public-facing product.
- The "exciting moments" detector is an audio-loudness heuristic, not a real
  scene-understanding model — it's fast and free, but it can occasionally
  pick a loud-but-boring moment (e.g. background noise) over a visually
  interesting but quiet one.
