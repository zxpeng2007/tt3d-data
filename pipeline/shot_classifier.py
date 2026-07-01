"""Standalone: classify sampled frames as 'gameplay' (table visible) or not.

Runs the upstream TT3D TableCalibrator, which returns a valid rotation only when
the regulation table is detected -> a robust proxy for the behind-the-table
broadcast (gameplay) camera vs. replays/crowd/close-ups.

Launch with cwd = <tt3d repo root> (so ./weights/table_segmentation.ckpt resolves)
and PYTHONPATH += <tt3d repo>/tt3d/calibration (bare imports). Prints JSON:
    {"<seconds>": true/false, ...}
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "tt3d", "calibration"))

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--timestamps", required=True, help="comma-separated seconds to sample")
    args = ap.parse_args()

    from table_calibrator import TableCalibrator

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(json.dumps({}))
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    calibrator = TableCalibrator(h, w)

    result = {}
    for ts in [float(x) for x in args.timestamps.split(",") if x != ""]:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ok, frame = cap.read()
        if not ok:
            result[str(ts)] = False
            continue
        try:
            rvec, tvec, f, er, _ = calibrator.process(frame, debug=True)
            result[str(ts)] = rvec is not None
        except Exception:
            result[str(ts)] = False
    cap.release()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
