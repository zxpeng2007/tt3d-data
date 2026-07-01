"""Standalone: classify sampled frames as 'gameplay' (table visible) or not.

Uses the upstream TT3D table segmentation model (via TableCalibrator.segment) and
flags a frame as gameplay when the table mask covers at least --threshold of the
frame. Mask presence is a far more robust gameplay signal than a full Hough+PnP
calibration, which succeeds only on sparse clean frames.

Frames are extracted with ffmpeg (not cv2.VideoCapture, which fails on the
Unicode characters yt-dlp puts in filenames on Windows).

Launch with cwd = <tt3d repo root> (so ./weights/table_segmentation.ckpt resolves)
and PYTHONPATH += <tt3d repo>/tt3d/calibration (bare imports). Prints JSON:
    {"<seconds>": true/false, ...}
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.getcwd(), "tt3d", "calibration"))

import cv2
import numpy as np


def _extract_frame(video: str, ts: float, out_png: str) -> bool:
    cmd = ["ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", video,
           "-frames:v", "1", "-q:v", "2", out_png]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0 and os.path.isfile(out_png) and os.path.getsize(out_png) > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--timestamps", required=True, help="comma-separated seconds to sample")
    ap.add_argument("--threshold", type=float, default=0.015,
                    help="min table-mask fraction of the frame to count as gameplay")
    args = ap.parse_args()

    from table_calibrator import TableCalibrator

    tss = [float(x) for x in args.timestamps.split(",") if x != ""]
    result = {}
    calibrator = None
    with tempfile.TemporaryDirectory() as tmp:
        for i, ts in enumerate(tss):
            png = os.path.join(tmp, f"f{i}.png")
            if not _extract_frame(args.video, ts, png):
                result[str(ts)] = False
                continue
            frame = cv2.imread(png)
            if frame is None:
                result[str(ts)] = False
                continue
            if calibrator is None:
                h, w = frame.shape[:2]
                calibrator = TableCalibrator(h, w)
            try:
                mask = calibrator.segment(frame)          # uint8 table mask
                frac = float((mask > 0).mean())
                result[str(ts)] = frac >= args.threshold
            except Exception:
                result[str(ts)] = False
    print(json.dumps(result))


if __name__ == "__main__":
    main()
