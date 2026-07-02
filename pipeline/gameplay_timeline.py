"""Standalone: build a gameplay timeline for a match video (fast, one pass).

Decodes only keyframes (``-skip_frame nokey`` -> fast) at a small resolution, then
runs the TT3D table segmenter on each and flags it as gameplay when the table mask
covers >= --threshold of the frame. Emits keyframe timestamps + gameplay booleans,
which segment_rallies merges into continuous gameplay segments.

Launch with cwd = <tt3d repo root> and PYTHONPATH += tt3d/calibration.
Output (stdout, JSON): {"times": [float, ...], "flags": [bool, ...]}
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.getcwd(), "tt3d", "calibration"))

import cv2  # noqa: E402

_PTS = re.compile(r"pts_time:([0-9.]+)")


def is_gameplay(mask, min_frac: float, max_frac: float, max_bbox_w: float = 0.55) -> bool:
    """Wide gameplay shot test on a table mask.

    Wide broadcast shot: the table is a small fraction of the frame, fully
    interior, and its bounding box spans only ~a quarter of the frame width.
    Close-up/replay: the table fills much of the frame, is clipped by the
    borders, or spans most of the frame width (measured 0.91 vs 0.26 on real
    replay vs wide frames).
    """
    import numpy as np
    m = mask > 0
    frac = float(m.mean())
    if frac < min_frac or frac > max_frac:
        return False
    b = 3  # border band (px at the downscaled size)
    if m[:, :b].mean() > 0.02 or m[:, -b:].mean() > 0.02 or m[-b:, :].mean() > 0.02:
        return False
    xs = np.where(m.any(axis=0))[0]
    if len(xs) and (xs[-1] - xs[0]) / m.shape[1] > max_bbox_w:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--threshold", type=float, default=0.02)
    ap.add_argument("--max-frac", type=float, default=0.15,
                    help="reject close-ups where the table exceeds this frame fraction")
    ap.add_argument("--scale", type=int, default=640)
    args = ap.parse_args()

    from table_calibrator import TableCalibrator

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            "ffmpeg", "-y", "-skip_frame", "nokey", "-i", args.video,
            "-vf", f"scale={args.scale}:-2,showinfo", "-an", "-vsync", "0",
            os.path.join(tmp, "%06d.png"),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              text=True, encoding="utf-8", errors="replace")
        times = [float(t) for t in _PTS.findall(proc.stderr or "")]
        pngs = sorted(glob.glob(os.path.join(tmp, "*.png")))
        n = min(len(times), len(pngs))
        times, pngs = times[:n], pngs[:n]

        cal = None
        flags = []
        for p in pngs:
            fr = cv2.imread(p)
            if fr is None:
                flags.append(False)
                continue
            if cal is None:
                cal = TableCalibrator(*fr.shape[:2])
            try:
                m = cal.segment(fr)
                flags.append(is_gameplay(m, args.threshold, args.max_frac))
            except Exception:
                flags.append(False)
    print(json.dumps({"times": times, "flags": flags}))


if __name__ == "__main__":
    main()
