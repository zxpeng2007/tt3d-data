"""Split full-match videos into gameplay rally clips.

Per match video:
  1. Build a gameplay timeline (pipeline/gameplay_timeline.py): decode keyframes
     once and flag each as gameplay when the table is visible.
  2. Merge consecutive gameplay keyframes into continuous segments (bridging brief
     cutaways up to `gameplay_merge_gap` seconds).
  3. Keep segments >= gameplay_min_seconds, split any longer than max_rally_seconds,
     and trim each to a rally clip.

Output: data/rallies/<match_id>/r0001.mp4 (+ r0001.json sidecar with provenance).

  python scripts/segment_rallies.py --videos data/videos --out data/rallies
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config
from pipeline.procutil import LOG, ffprobe
from pipeline.video import extract_clip


def gameplay_timeline(video: Path, threshold: float, max_frac: float) -> tuple[list[float], list[bool]]:
    """Run the keyframe gameplay classifier; return (keyframe_times, gameplay_flags)."""
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONUTF8": "1"}
    calib = config.TT3D_DIR / "tt3d" / "calibration"
    env["PYTHONPATH"] = os.pathsep.join([str(calib), env.get("PYTHONPATH", "")])
    cmd = [config.PYTHON,
           str((config.REPO_ROOT / "pipeline" / "gameplay_timeline.py").resolve()),
           "--video", str(video), "--threshold", str(threshold),
           "--max-frac", str(max_frac)]
    proc = subprocess.run(cmd, cwd=str(config.TT3D_DIR), env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, encoding="utf-8", errors="replace")
    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
        return data["times"], data["flags"]
    except Exception:
        LOG.warning("timeline parse failed for %s: %s", video.name, (proc.stderr or "")[-300:])
        return [], []


def merge_segments(times: list[float], flags: list[bool], max_gap: float,
                   min_dur: float, pad: float) -> list[tuple[float, float]]:
    """Merge gameplay keyframes into [start, end] segments (seconds)."""
    gt = [t for t, f in zip(times, flags) if f]
    if not gt:
        return []
    segs: list[list[float]] = [[gt[0], gt[0]]]
    for t in gt[1:]:
        if t - segs[-1][1] <= max_gap:
            segs[-1][1] = t
        else:
            segs.append([t, t])
    out = []
    for s, e in segs:
        s2, e2 = max(0.0, s - pad), e + pad
        if e2 - s2 >= min_dur:
            out.append((s2, e2))
    return out


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
    if info["duration"] <= 0:
        LOG.warning("skip %s (no duration)", video.name)
        return 0
    match_id, source_url = _match_id(video)
    out_dir = out_root / match_id
    out_dir.mkdir(parents=True, exist_ok=True)

    times, flags = gameplay_timeline(video, cfg.table_presence_min, cfg.table_presence_max)
    n_game = sum(1 for f in flags if f)
    LOG.info("%s: %d keyframes, %d gameplay", match_id, len(flags), n_game)

    segments = merge_segments(times, flags,
                              max_gap=cfg.gameplay_merge_gap,
                              min_dur=cfg.gameplay_min_seconds, pad=0.5)
    LOG.info("%s: %d gameplay segments", match_id, len(segments))

    n = 0
    for s, e in segments:
        dur = e - s
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
    videos = [p.resolve() for p in sorted(args.videos.glob(args.glob))
              if p.is_file() and ".f298" not in p.name and ".part" not in p.name]
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
