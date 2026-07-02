"""Repair the 90-deg camera orientation on already-reconstructed rallies.

For each rally dir with camera.yaml + mb_input.json, run the reorientation
check; when the camera is rewritten, regenerate the camera-dependent outputs
(world-frame poses + 3D ball). GPU-heavy artefacts (2D pose, MotionBERT
camera-frame output, BlurBall 2D detections) do not depend on the camera and
are reused.

  python scripts/repair_orientation.py --dataset data/dataset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import ball, body, config, reorient
from pipeline.procutil import LOG


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=config.DATASET_DIR)
    args = ap.parse_args()
    cfg = config.load_config()

    rally_dirs = sorted(p.parent for p in (args.dataset / "rallies").glob("*/*/camera.yaml"))
    LOG.info("checking %d rallies", len(rally_dirs))
    fixed = 0
    for d in rally_dirs:
        if not reorient.check_and_fix(d):
            continue
        fixed += 1
        # regenerate camera-dependent outputs
        for f in ("p0_3d.npy", "p1_3d.npy", "ball_traj_3D.csv"):
            (d / f).unlink(missing_ok=True)
        try:
            body.run_body(d, cfg, force=True)
        except Exception as exc:
            LOG.warning("[repair] body failed for %s: %s", d.name, exc)
        try:
            ball.run_ball(d, cfg)
        except Exception as exc:
            LOG.warning("[repair] ball failed for %s: %s", d.name, exc)
    LOG.info("repair done: %d/%d rallies reoriented", fixed, len(rally_dirs))


if __name__ == "__main__":
    main()
