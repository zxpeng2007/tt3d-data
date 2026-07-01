"""Aggregate all reconstructed rallies into a manifest + dataset card.

Scans <dataset>/rallies/**/meta.json and writes:
  <dataset>/manifest.parquet   one row per rally (see docs/DATASET_SCHEMA.md)
  <dataset>/dataset_card.json   totals and quality summary

  python scripts/aggregate_dataset.py --dataset data/dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from pipeline import config


COLUMNS = [
    "match_id", "rally_id", "source_url", "n_frames", "fps", "width", "height",
    "calib_ok", "has_ball_2d", "has_ball_3d", "has_pose_p0", "has_pose_p1",
    "ball_coverage", "ok", "path",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=config.DATASET_DIR)
    args = ap.parse_args()

    rallies_root = args.dataset / "rallies"
    rows = []
    for meta_path in rallies_root.glob("**/meta.json"):
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        q = m.get("quality", {})
        rows.append({
            "match_id": m.get("match_id"),
            "rally_id": m.get("rally_id"),
            "source_url": m.get("source_url"),
            "n_frames": m.get("n_frames", 0),
            "fps": m.get("fps", 0.0),
            "width": m.get("width", 0),
            "height": m.get("height", 0),
            "calib_ok": bool(q.get("calib_ok", m.get("calib_ok", False))),
            "has_ball_2d": bool(q.get("has_ball_2d", False)),
            "has_ball_3d": bool(q.get("has_ball_3d", False)),
            "has_pose_p0": bool(q.get("has_pose_p0", False)),
            "has_pose_p1": bool(q.get("has_pose_p1", False)),
            "ball_coverage": float(q.get("ball_coverage", 0.0)),
            "ok": bool(m.get("ok", False)),
            "path": str(meta_path.parent.relative_to(args.dataset)),
        })

    df = pd.DataFrame(rows, columns=COLUMNS)
    manifest = args.dataset / "manifest.parquet"
    df.to_parquet(manifest, index=False)

    card = {
        "n_rallies": int(len(df)),
        "n_ok": int(df["ok"].sum()) if len(df) else 0,
        "n_matches": int(df["match_id"].nunique()) if len(df) else 0,
        "total_frames": int(df["n_frames"].sum()) if len(df) else 0,
        "with_ball_2d": int(df["has_ball_2d"].sum()) if len(df) else 0,
        "with_ball_3d": int(df["has_ball_3d"].sum()) if len(df) else 0,
        "with_both_poses": int((df["has_pose_p0"] & df["has_pose_p1"]).sum()) if len(df) else 0,
        "mean_ball_coverage": round(float(df.loc[df["ok"], "ball_coverage"].mean()), 3)
            if len(df) and df["ok"].any() else 0.0,
    }
    (args.dataset / "dataset_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    print(json.dumps(card, indent=2))
    print(f"\nWrote {manifest} ({len(df)} rows)")


if __name__ == "__main__":
    main()
