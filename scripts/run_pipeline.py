"""Run the full reconstruction on a single rally clip (validation / one-off).

Examples:
  # Reconstruct one clip into a rally dir
  python scripts/run_pipeline.py --clip data/rallies/<match>/r0001.mp4 \
      --out data/dataset/rallies/<match>/r0001

  # Re-run only some stages on an existing rally dir
  python scripts/run_pipeline.py --rally-dir <dir> --stages table body
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config
from pipeline.rally_pipeline import ALL_STAGES, process_rally


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", type=Path, help="Source rally clip (mp4)")
    ap.add_argument("--rally-dir", type=Path, help="Rally working dir (reuses existing rally.mp4)")
    ap.add_argument("--out", type=Path, help="Output rally dir (with --clip)")
    ap.add_argument("--stages", nargs="+", default=list(ALL_STAGES),
                    choices=list(ALL_STAGES), help="Subset of stages to run")
    ap.add_argument("--force", action="store_true", help="Recompute even if outputs exist")
    args = ap.parse_args()

    cfg = config.load_config()
    config.ensure_dirs()

    if args.rally_dir and not args.clip:
        clip = args.rally_dir / "rally.mp4"
        out = args.rally_dir
    elif args.clip:
        clip = args.clip
        out = args.out or (config.DATASET_DIR / "rallies" / clip.stem)
    else:
        ap.error("provide --clip or --rally-dir")

    res = process_rally(clip, out, cfg, stages=tuple(args.stages), force=args.force)
    print("\n=== result ===")
    for k, v in res.stages.items():
        print(f"  {k:10s}: {v}")
    print(f"  ok={res.ok} frames={res.n_frames} ball_cov={res.ball_coverage:.2f} "
          f"pose={res.has_pose_p0}/{res.has_pose_p1} ball3d={res.has_ball_3d}")
    sys.exit(0 if res.ok else 2)


if __name__ == "__main__":
    main()
