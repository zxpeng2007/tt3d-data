"""Split full-match videos into gameplay rally clips.

Pipeline per match video:
  1. Detect shot boundaries with ffmpeg scene detection (on a downscaled stream).
  2. Classify each shot's mid-frame as gameplay (table visible) using the upstream
     TableCalibrator via pipeline/shot_classifier.py.
  3. Keep gameplay shots >= gameplay_min_seconds, split any longer than
     max_rally_seconds into chunks, and trim each to a rally clip.

Output: data/rallies/<match_id>/r0001.mp4 (+ r0001.json sidecar with provenance).

  python scripts/segment_rallies.py --videos data/videos --out data/rallies
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config
from pipeline.procutil import LOG, ffprobe
from pipeline.video import extract_clip

_PTS = re.compile(r"pts_time:([0-9.]+)")


def detect_cuts(video: Path, threshold: float) -> list[float]:
    """Return sorted shot-boundary timestamps (seconds) via ffmpeg scene detection."""
    cmd = [
        "ffmpeg", "-hide_banner", "-i", str(video),
        "-filter:v", f"scale=320:-2,select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                          text=True, encoding="utf-8", errors="replace")
    times = sorted({float(m) for m in _PTS.findall(proc.stderr or "")})
    return times


def classify(video: Path, timestamps: list[float], threshold: float) -> dict[float, bool]:
    """Classify sampled timestamps as gameplay (table mask present) via subprocess."""
    if not timestamps:
        return {}
    import os
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONUTF8": "1"}
    calib = config.TT3D_DIR / "tt3d" / "calibration"
    env["PYTHONPATH"] = os.pathsep.join([str(calib), env.get("PYTHONPATH", "")])
    ts_arg = ",".join(f"{t:.3f}" for t in timestamps)
    cmd = [config.PYTHON, str((config.REPO_ROOT / "pipeline" / "shot_classifier.py").resolve()),
           "--video", str(video), "--timestamps", ts_arg, "--threshold", str(threshold)]
    proc = subprocess.run(cmd, cwd=str(config.TT3D_DIR), env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, encoding="utf-8", errors="replace")
    try:
        raw = json.loads(proc.stdout.strip().splitlines()[-1])
        return {float(k): bool(v) for k, v in raw.items()}
    except Exception:
        LOG.warning("classify parse failed for %s: %s", video.name, proc.stderr[-300:])
        return {}


def _match_id(video: Path) -> tuple[str, str | None]:
    info = video.with_suffix(".info.json")
    if info.exists():
        try:
            d = json.loads(info.read_text(encoding="utf-8"))
            return d.get("id", video.stem), d.get("webpage_url")
        except Exception:
            pass
    m = re.search(r"\[([A-Za-z0-9_-]{6,})\]", video.stem)
    return (m.group(1) if m else video.stem), None


def segment_match(video: Path, out_root: Path, cfg: config.PipelineConfig) -> int:
    info = ffprobe(video)
    duration = info["duration"]
    if duration <= 0:
        LOG.warning("skip %s (no duration)", video.name)
        return 0
    match_id, source_url = _match_id(video)
    out_dir = out_root / match_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cuts = detect_cuts(video, cfg.scene_cut_threshold)
    bounds = [0.0, *cuts, duration]
    shots = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
             if bounds[i + 1] - bounds[i] >= 1.0]
    LOG.info("%s: %.0fs, %d shots", match_id, duration, len(shots))

    mids = [(s + e) / 2.0 for s, e in shots]
    flags = classify(video, mids, cfg.table_presence_min)
    n_game = sum(1 for mid in mids if flags.get(round(mid, 3), flags.get(mid, False)))
    LOG.info("%s: %d/%d shots classified gameplay", match_id, n_game, len(shots))

    n = 0
    for (s, e), mid in zip(shots, mids):
        if not flags.get(round(mid, 3), flags.get(mid, False)):
            continue
        dur = e - s
        if dur < cfg.gameplay_min_seconds:
            continue
        n_chunks = max(1, math.ceil(dur / cfg.max_rally_seconds))
        chunk = dur / n_chunks
        for c in range(n_chunks):
            cs, ce = s + c * chunk, s + (c + 1) * chunk
            n += 1
            clip = out_dir / f"r{n:04d}.mp4"
            if clip.exists():
                continue
            extract_clip(video, clip, cs, ce, reencode=True)
            clip.with_suffix(".json").write_text(json.dumps({
                "match_id": match_id, "rally_id": clip.stem,
                "source_url": source_url, "source_video": video.name,
                "start_sec": round(cs, 3), "end_sec": round(ce, 3),
            }, indent=2), encoding="utf-8")
    LOG.info("%s: wrote %d rally clips", match_id, n)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", type=Path, default=config.VIDEOS_DIR)
    ap.add_argument("--out", type=Path, default=config.RALLIES_DIR)
    ap.add_argument("--glob", default="**/*.mp4")
    args = ap.parse_args()

    cfg = config.load_config()
    config.ensure_dirs()
    videos = [p for p in sorted(args.videos.glob(args.glob)) if p.is_file()]
    if not videos:
        LOG.warning("no videos under %s", args.videos)
        return
    total = 0
    for v in videos:
        try:
            total += segment_match(v, args.out, cfg)
        except Exception:
            LOG.exception("segmentation failed for %s", v.name)
    LOG.info("Done: %d rally clips from %d matches", total, len(videos))


if __name__ == "__main__":
    main()
