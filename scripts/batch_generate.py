"""Batch reconstruction over every rally clip -> dataset (resumable).

Walks a directory of rally clips (produced by segment_rallies.py), laid out as
    <rallies>/<match_id>/<rally_id>.mp4
and reconstructs each into
    <out>/rallies/<match_id>/<rally_id>/
skipping any rally whose meta.json already reports ok (unless --force).

A JSONL progress log is appended at <out>/progress.jsonl so long runs can be
monitored and safely resumed after interruption.

Examples:
  python scripts/batch_generate.py --rallies data/rallies --out data/dataset
  python scripts/batch_generate.py --rallies data/rallies --out data/dataset --limit 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config
from pipeline.procutil import LOG
from pipeline.rally_pipeline import ALL_STAGES, process_rally


def _iter_clips(rallies_dir: Path):
    for match_dir in sorted(p for p in rallies_dir.iterdir() if p.is_dir()):
        for clip in sorted(match_dir.glob("*.mp4")):
            yield match_dir.name, clip


def _already_done(out_rally: Path) -> bool:
    meta = out_rally / "meta.json"
    if not meta.exists():
        return False
    try:
        return bool(json.loads(meta.read_text(encoding="utf-8")).get("ok"))
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rallies", type=Path, default=config.RALLIES_DIR)
    ap.add_argument("--out", type=Path, default=config.DATASET_DIR)
    ap.add_argument("--stages", nargs="+", default=list(ALL_STAGES), choices=list(ALL_STAGES))
    ap.add_argument("--limit", type=int, default=0, help="Max rallies to process (0 = all)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = config.load_config()
    config.ensure_dirs()
    out_rallies = args.out / "rallies"
    out_rallies.mkdir(parents=True, exist_ok=True)
    progress = args.out / "progress.jsonl"

    clips = list(_iter_clips(args.rallies))
    if not clips:
        LOG.warning("No rally clips found under %s", args.rallies)
        return
    LOG.info("Found %d rally clips under %s", len(clips), args.rallies)

    done = ok = failed = 0
    t0 = time.time()
    for match_id, clip in clips:
        rally_id = clip.stem
        out_rally = out_rallies / match_id / rally_id
        if _already_done(out_rally) and not args.force:
            done += 1
            continue

        sidecar = clip.with_suffix(".json")
        meta_extra = {}
        if sidecar.exists():
            try:
                meta_extra = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:
                pass
        meta_extra.setdefault("match_id", match_id)
        meta_extra.setdefault("rally_id", rally_id)

        LOG.info("[%d/%d] %s/%s", done + 1, len(clips), match_id, rally_id)
        try:
            res = process_rally(clip, out_rally, cfg, meta_extra=meta_extra,
                                stages=tuple(args.stages), force=args.force)
            ok += int(res.ok)
            failed += int(not res.ok)
            record = {"match_id": match_id, "rally_id": rally_id, **asdict(res)}
        except Exception as exc:  # never let one rally kill the batch
            failed += 1
            LOG.exception("rally %s/%s crashed", match_id, rally_id)
            record = {"match_id": match_id, "rally_id": rally_id, "ok": False, "error": str(exc)}

        with open(progress, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        done += 1
        if args.limit and (ok + failed) >= args.limit:
            LOG.info("Reached --limit %d", args.limit)
            break

    dt = time.time() - t0
    LOG.info("Batch done: %d processed, %d ok, %d failed, %.1f min",
             ok + failed, ok, failed, dt / 60.0)


if __name__ == "__main__":
    main()
