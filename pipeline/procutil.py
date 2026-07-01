"""Subprocess, logging and ffprobe helpers used across pipeline stages."""
from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

# This machine's console/locale is GBK (Chinese Windows). Force UTF-8 on our
# stdio so logging filenames/output with non-GBK characters never crash.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def get_logger(name: str = "tt3d-data") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


LOG = get_logger()


class StageError(RuntimeError):
    """Raised when an external pipeline stage fails."""


def run(
    cmd: list[str],
    cwd: Path | str | None = None,
    env: dict | None = None,
    log_path: Path | None = None,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Run a command, streaming combined output to `log_path` (and capturing it).

    Upstream scripts are noisy and some (rally.py) call matplotlib; callers pass an
    env with MPLBACKEND=Agg to make plt.show() a no-op. Returns the completed
    process; raises StageError on non-zero exit when check=True.
    """
    cmd = [str(c) for c in cmd]
    full_env = {**os.environ, **(env or {})}
    LOG.info("+ %s  (cwd=%s)", " ".join(shlex.quote(c) for c in cmd), cwd or ".")
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(proc.stdout or "", encoding="utf-8", errors="replace")
    if check and proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-40:])
        raise StageError(
            f"Command failed (exit {proc.returncode}): {' '.join(cmd)}\n--- last 40 lines ---\n{tail}"
        )
    return proc


def agg_env() -> dict:
    """Environment that forces a non-interactive matplotlib backend."""
    return {"MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}


def ffprobe(video: Path | str) -> dict:
    """Return {fps, n_frames, width, height, duration} for a video via ffprobe."""
    video = str(video)
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate,nb_frames,width,height,duration",
        "-show_entries", "format=duration",
        "-of", "json", video,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise StageError(f"ffprobe failed for {video}: {proc.stderr}")
    info = json.loads(proc.stdout)
    stream = (info.get("streams") or [{}])[0]

    def _rate(val: str) -> float:
        try:
            num, den = val.split("/")
            return float(num) / float(den) if float(den) else 0.0
        except Exception:
            return 0.0

    fps = _rate(stream.get("avg_frame_rate", "0/0")) or _rate(stream.get("r_frame_rate", "0/0"))
    duration = float(stream.get("duration") or info.get("format", {}).get("duration") or 0.0)
    nb_frames = stream.get("nb_frames")
    try:
        n_frames = int(nb_frames)
    except (TypeError, ValueError):
        n_frames = int(round(fps * duration)) if fps and duration else 0
    return {
        "fps": fps,
        "n_frames": n_frames,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "duration": duration,
    }
