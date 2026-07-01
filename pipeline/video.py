"""Video preparation: normalize a rally clip into a canonical, de-duplicated mp4.

Broadcast footage is often re-encoded at a container fps that differs from the
capture fps, producing duplicate frames. TT3D's ball/pose/camera stages assume a
clean 1:1 frame index at a known fps (rally.py hardcodes 25). We therefore render
every rally to a canonical clip: constant CANONICAL_FPS, duplicate frames dropped
(mpdecimate), so ball_traj_2D / poses / camera all share the same frame index.
"""
from __future__ import annotations

from pathlib import Path

from . import config
from .procutil import LOG, StageError, ffprobe, run


def make_canonical_rally(
    src: Path | str,
    dst: Path | str,
    fps: int | None = None,
    dedup: bool = True,
    crf: int = 18,
) -> dict:
    """Re-encode `src` to a canonical `dst` mp4 at fixed fps with duplicate frames removed.

    Returns the ffprobe info of the resulting clip. Idempotent: if `dst` already
    exists and is non-empty, it is kept and just probed.
    """
    src, dst = Path(src), Path(dst)
    fps = fps or config.CANONICAL_FPS
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and dst.stat().st_size > 0:
        return ffprobe(dst)

    # mpdecimate drops near-duplicate frames; the following fps filter resamples to
    # a constant rate. Order matters: decimate first, then set the constant fps.
    vf = "mpdecimate,setpts=N/FRAME_RATE/TB" if dedup else "null"
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"{vf},fps={fps}",
        "-an",  # drop audio; not needed
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(dst),
    ]
    run(cmd, log_path=config.LOGS_DIR / "ffmpeg_canonical.log")
    if not dst.exists() or dst.stat().st_size == 0:
        raise StageError(f"Canonical re-encode produced no output: {dst}")
    info = ffprobe(dst)
    LOG.info("canonical rally %s: %d frames @ %.2f fps (%.1fs)",
             dst.name, info["n_frames"], info["fps"], info["duration"])
    return info


def extract_clip(
    src: Path | str,
    dst: Path | str,
    start_sec: float,
    end_sec: float,
    reencode: bool = True,
) -> Path:
    """Trim [start_sec, end_sec) from `src` into `dst`.

    Uses re-encoding (accurate cuts) by default; set reencode=False for a fast,
    keyframe-aligned copy when frame accuracy is not required.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.0, end_sec - start_sec)
    if reencode:
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start_sec:.3f}", "-i", str(src),
            "-t", f"{dur:.3f}", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", str(dst),
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start_sec:.3f}", "-i", str(src),
            "-t", f"{dur:.3f}", "-an", "-c", "copy", str(dst),
        ]
    run(cmd, log_path=config.LOGS_DIR / "ffmpeg_clip.log")
    return dst
