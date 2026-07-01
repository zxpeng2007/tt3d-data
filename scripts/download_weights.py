"""Fetch model weights for the pipeline.

  table segmentation : ships inside third_party/tt3d/weights (verified here)
  BlurBall           : downloaded from the BlurBall Nextcloud share
  MotionBERT (lite)  : checkpoint for infer (OneDrive; HF mirror attempted)
  RTMPose/RTMDet     : auto-downloaded by rtmlib on first use (no action)

Env overrides for robustness:
  BLURBALL_URL   direct URL to the BlurBall weight file
  MB_CKPT_URL    direct URL to best_epoch.bin

  python scripts/download_weights.py
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from pipeline import config

BLURBALL_SHARE = "https://cloud.cs.uni-tuebingen.de/index.php/s/6Z8TpM3sXRKHzGC"
BLURBALL_TOKEN = "6Z8TpM3sXRKHzGC"
BLURBALL_FILE = "blurball_best"  # exact file in the Nextcloud share (~6 MB torch ckpt)
MB_ONEDRIVE_NOTE = (
    "MotionBERT checkpoint 'FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin'.\n"
    "  Get it from the MotionBERT repo's model zoo (OneDrive link in its README:\n"
    "  https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md ), then place at:\n"
    f"  {config.MOTIONBERT_CKPT}\n"
    "  Or set MB_CKPT_URL to a direct link and re-run."
)


def _download(url: str, dst: Path, desc: str) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[weights] downloading {desc} ...")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            got = 0
            with open(dst, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    got += len(chunk)
                    if total:
                        print(f"\r  {got/1e6:6.1f}/{total/1e6:6.1f} MB", end="", flush=True)
            print()
        return True
    except Exception as exc:
        print(f"[weights] FAILED {desc}: {exc}")
        return False


def check_table_seg() -> bool:
    ok = config.TABLE_SEG_CKPT.exists()
    print(f"[weights] table segmentation: {'OK ' + str(config.TABLE_SEG_CKPT) if ok else 'MISSING'}")
    if not ok:
        print("  It ships with upstream TT3D; ensure third_party/tt3d is checked out.")
    return ok


def fetch_blurball() -> bool:
    if config.BLURBALL_CKPT.exists():
        print(f"[weights] BlurBall: OK {config.BLURBALL_CKPT}")
        return True
    url = os.environ.get("BLURBALL_URL")
    dst = config.BLURBALL_CKPT
    dst.parent.mkdir(parents=True, exist_ok=True)
    if url:
        return _download(url, dst, "BlurBall weight")
    # Direct WebDAV file download from the public Nextcloud share (token as user).
    dav = f"https://cloud.cs.uni-tuebingen.de/public.php/dav/files/{BLURBALL_TOKEN}/{BLURBALL_FILE}"
    print(f"[weights] BlurBall: fetching {BLURBALL_FILE} from Nextcloud share ...")
    try:
        with requests.get(dav, auth=(BLURBALL_TOKEN, ""), stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dst, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        print(f"[weights] BlurBall: saved -> {dst}")
        return True
    except Exception as exc:
        print(f"[weights] BlurBall FAILED: {exc}\n  Download '{BLURBALL_FILE}' from {BLURBALL_SHARE} and place at {dst}")
        return False


def fetch_motionbert() -> bool:
    if config.MOTIONBERT_CKPT.exists():
        print(f"[weights] MotionBERT: OK {config.MOTIONBERT_CKPT}")
        return True
    url = os.environ.get("MB_CKPT_URL") or (
        "https://huggingface.co/walterzhu/MotionBERT/resolve/main/"
        "checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin"
    )
    if _download(url, config.MOTIONBERT_CKPT, "MotionBERT lite checkpoint (HF)"):
        return True
    print("[weights] MotionBERT: automatic download failed.\n  " + MB_ONEDRIVE_NOTE)
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-blurball", action="store_true")
    ap.add_argument("--skip-motionbert", action="store_true")
    args = ap.parse_args()
    config.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {"table_seg": check_table_seg()}
    if not args.skip_blurball:
        results["blurball"] = fetch_blurball()
    if not args.skip_motionbert:
        results["motionbert"] = fetch_motionbert()
    print("\n[weights] RTMPose/RTMDet: auto-downloaded by rtmlib on first inference.")
    print("\n=== summary ===")
    for k, v in results.items():
        print(f"  {k:12s}: {'ready' if v else 'MISSING (see notes above)'}")


if __name__ == "__main__":
    main()
