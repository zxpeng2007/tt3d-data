"""End-to-end bulk driver: download -> segment -> reconstruct -> aggregate.

Runs the four stages in sequence; every stage is resumable, so this can be
re-run (or looped) safely to grow the dataset over time.

  python scripts/run_bulk.py                      # all configured sources
  python scripts/run_bulk.py --source ittf-worlds-doha-2025 --limit 20
  python scripts/run_bulk.py --skip-download      # only (re)process what's on disk
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pipeline import config  # noqa: E402
from pipeline.procutil import LOG  # noqa: E402

PY = config.PYTHON
S = REPO / "scripts"


def step(name: str, cmd: list[str]) -> None:
    LOG.info("=== %s ===", name)
    rc = subprocess.run([str(c) for c in cmd]).returncode
    if rc not in (0, 101):
        LOG.warning("step '%s' exited %d (continuing)", name, rc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", help="restrict to a single configured source")
    ap.add_argument("--limit", type=int, default=0, help="max matches to download")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-segment", action="store_true")
    args = ap.parse_args()

    if not args.skip_download:
        cmd = [PY, S / "download_videos.py"]
        if args.source:
            cmd += ["--source", args.source]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        step("download", cmd)

    if not args.skip_segment:
        step("segment", [PY, S / "segment_rallies.py",
                         "--videos", config.VIDEOS_DIR, "--out", config.RALLIES_DIR])

    step("reconstruct", [PY, S / "batch_generate.py",
                         "--rallies", config.RALLIES_DIR, "--out", config.DATASET_DIR])
    step("aggregate", [PY, S / "aggregate_dataset.py", "--dataset", config.DATASET_DIR])
    LOG.info("Bulk pass complete. Re-run to extend as more matches download.")


if __name__ == "__main__":
    main()
