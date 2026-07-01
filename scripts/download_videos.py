"""Bulk-download WTT/ITTF full matches with yt-dlp (resumable).

Reads configs/sources.yaml, filters to "FULL MATCH" uploads, and downloads them
into data/videos/<source>/ with a --download-archive so re-runs skip completed
files. An .info.json sidecar is written per video (source url / id / title) for
downstream provenance.

Examples:
  # See what would be downloaded from one source (no download)
  python scripts/download_videos.py --source ittf-worlds-doha-2025 --list

  # Download up to 100 full matches across all configured sources
  python scripts/download_videos.py --limit 100

  # Download one playlist at 1080p
  python scripts/download_videos.py --source wtt-us-smash-2026 --max-height 1080
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from pipeline import config

YTDLP = [config.PYTHON, "-m", "yt_dlp"]


def _sources(cfg: dict) -> list[dict]:
    out = []
    for c in cfg.get("channels", []):
        out.append({"kind": "channel", **c})
    for p in cfg.get("playlists", []):
        out.append({"kind": "playlist", **p})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=config.REPO_ROOT / "configs" / "sources.yaml")
    ap.add_argument("--source", help="Only this named source (channel/playlist)")
    ap.add_argument("--limit", type=int, default=0, help="Max videos to download (0 = all)")
    ap.add_argument("--max-height", type=int, help="Override max video height (e.g. 1080)")
    ap.add_argument("--list", action="store_true", help="List matches only, no download")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    max_h = args.max_height or cfg.get("max_height", 720)
    fmt = cfg["format"].replace("{max_height}", str(max_h))
    match_filter = cfg["match_filter"]
    archive = config.VIDEOS_DIR / "archive.txt"
    config.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    sources = _sources(cfg)
    if args.source:
        sources = [s for s in sources if s["name"] == args.source]
        if not sources:
            ap.error(f"unknown source '{args.source}'")

    for s in sources:
        out_dir = config.VIDEOS_DIR / s["name"]
        if args.list:
            cmd = [*YTDLP, "--flat-playlist", "--match-filter", match_filter,
                   "--print", "%(duration>%H:%M:%S)s | %(title)s | https://youtu.be/%(id)s", s["url"]]
        else:
            cmd = [
                *YTDLP,
                "--match-filter", match_filter,
                "-f", fmt,
                "--merge-output-format", cfg.get("merge_output_format", "mp4"),
                "--download-archive", str(archive),
                "--write-info-json",
                "--no-overwrites",
                "--concurrent-fragments", "8",
                "--retries", "10", "--fragment-retries", "10",
                "--sleep-requests", str(cfg.get("sleep_requests", 1)),
                "--sleep-interval", str(cfg.get("sleep_interval", 5)),
                "--max-sleep-interval", str(cfg.get("max_sleep_interval", 15)),
                "-o", str(out_dir / "%(title).80s [%(id)s].%(ext)s"),
                s["url"],
            ]
            if args.limit:
                cmd[1:1] = ["--max-downloads", str(args.limit)]
        print(f"\n=== {s['kind']} {s['name']} ===", flush=True)
        rc = subprocess.run([str(c) for c in cmd]).returncode
        # yt-dlp returns 101 when --max-downloads limit is reached: not an error.
        if rc not in (0, 101):
            print(f"yt-dlp exited {rc} for {s['name']}", file=sys.stderr)


if __name__ == "__main__":
    main()
