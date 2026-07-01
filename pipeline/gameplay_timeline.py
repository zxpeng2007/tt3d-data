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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--threshold", type=float, default=0.02)
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
                flags.append(bool(float((m > 0).mean()) >= args.threshold))
            except Exception:
                flags.append(False)
    print(json.dumps({"times": times, "flags": flags}))


if __name__ == "__main__":
    main()
