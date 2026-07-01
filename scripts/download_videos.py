"""Download WTT/ITTF SINGLES full matches with a per-player footage budget.

Two-pass, resumable:
  1. List candidate videos across the configured sources (fast, no download) and
     keep only SINGLES full matches (title "FULL MATCH", not doubles).
  2. Greedily select matches so that no player exceeds --max-hours-per-player of
     raw footage (default 5 h), then download the selected videos.

Player names are parsed from the "PlayerA vs PlayerB" segment of the title; the
budget caps how much footage any single player contributes, keeping the raw
dataset small and diverse.

  python scripts/download_videos.py                       # all sources, 5h/player
  python scripts/download_videos.py --max-hours-per-player 5 --dry-run
  python scripts/download_videos.py --source ittf-worlds-doha-2025
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from pipeline import config

YTDLP = [config.PYTHON, "-m", "yt_dlp"]
DOUBLES = re.compile(r"\b[MWX]D\b|/", re.IGNORECASE)


def _sources(cfg: dict, only: str | None) -> list[dict]:
    out = [{"kind": "channel", **c} for c in cfg.get("channels", [])]
    out += [{"kind": "playlist", **p} for p in cfg.get("playlists", [])]
    if only:
        out = [s for s in out if s["name"] == only]
    return out


def parse_players(title: str) -> tuple[str, str] | None:
    """Extract (playerA, playerB) from a 'FULL MATCH | A vs B | ...' title."""
    parts = [p.strip() for p in title.split("|")]
    seg = next((p for p in parts if re.search(r"\bvs\b", p, re.IGNORECASE)), None)
    if not seg:
        return None
    m = re.split(r"\s+vs\.?\s+", seg, flags=re.IGNORECASE)
    if len(m) != 2:
        return None
    a, b = m[0].strip(), m[1].strip()
    if not a or not b:
        return None
    return a, b


def list_candidates(source: dict) -> list[dict]:
    """Return [{id, duration, title, players}] singles full matches for a source."""
    cmd = [*YTDLP, "--flat-playlist", "--print", "%(id)s\t%(duration)s\t%(title)s", source["url"]]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          text=True, encoding="utf-8", errors="replace")
    cands = []
    for line in (proc.stdout or "").splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        vid, dur, title = parts
        if "full match" not in title.lower():
            continue
        try:
            dur_s = float(dur)
        except ValueError:
            continue
        if dur_s < 600:                       # skip teasers/highlights
            continue
        if DOUBLES.search(title):             # singles only
            continue
        players = parse_players(title)
        if not players:
            continue
        cands.append({"id": vid, "duration": dur_s, "title": title,
                      "players": players, "source": source["name"]})
    return cands


def select_within_budget(cands: list[dict], budget_s: float) -> list[dict]:
    """Greedily pick matches so no player exceeds budget_s of footage."""
    spent: dict[str, float] = {}
    picked = []
    for c in cands:
        a, b = c["players"]
        if spent.get(a, 0.0) >= budget_s or spent.get(b, 0.0) >= budget_s:
            continue
        picked.append(c)
        spent[a] = spent.get(a, 0.0) + c["duration"]
        spent[b] = spent.get(b, 0.0) + c["duration"]
    return picked, spent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=config.REPO_ROOT / "configs" / "sources.yaml")
    ap.add_argument("--source", help="restrict to one configured source")
    ap.add_argument("--max-hours-per-player", type=float, default=5.0)
    ap.add_argument("--max-height", type=int)
    ap.add_argument("--dry-run", action="store_true", help="show selection, don't download")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    max_h = args.max_height or cfg.get("max_height", 720)
    fmt = cfg["format"].replace("{max_height}", str(max_h))
    budget_s = args.max_hours_per_player * 3600.0
    config.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    archive = config.VIDEOS_DIR / "archive.txt"

    cands: list[dict] = []
    for s in _sources(cfg, args.source):
        found = list_candidates(s)
        print(f"[{s['name']}] {len(found)} singles full matches", flush=True)
        cands += found

    picked, spent = select_within_budget(cands, budget_s)
    total_h = sum(c["duration"] for c in picked) / 3600.0
    print(f"\nSelected {len(picked)}/{len(cands)} matches, {total_h:.1f} h total, "
          f"{len(spent)} players (cap {args.max_hours_per_player} h each):")
    for p, sec in sorted(spent.items(), key=lambda kv: -kv[1]):
        print(f"  {sec/3600:4.1f} h  {p}")

    if args.dry_run:
        return

    urls = [f"https://youtu.be/{c['id']}" for c in picked]
    if not urls:
        print("Nothing to download.")
        return
    cmd = [
        *YTDLP, "-f", fmt,
        "--merge-output-format", cfg.get("merge_output_format", "mp4"),
        "--download-archive", str(archive),
        "--write-info-json", "--no-overwrites", "--restrict-filenames",
        "--concurrent-fragments", "8", "--retries", "10", "--fragment-retries", "10",
        "--sleep-requests", str(cfg.get("sleep_requests", 1)),
        "--sleep-interval", str(cfg.get("sleep_interval", 5)),
        "--max-sleep-interval", str(cfg.get("max_sleep_interval", 15)),
        "-o", str(config.VIDEOS_DIR / "%(id)s" / "%(title).80s [%(id)s].%(ext)s"),
        *urls,
    ]
    rc = subprocess.run([str(c) for c in cmd]).returncode
    if rc not in (0, 101):
        print(f"yt-dlp exited {rc}", file=sys.stderr)


if __name__ == "__main__":
    main()
