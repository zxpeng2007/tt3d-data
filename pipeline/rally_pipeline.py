"""Single-rally orchestration: video -> table, body, ball -> meta.json.

Each stage is resumable (skips when its output exists) and independently guarded,
so a failure in one stage (e.g. ball) still preserves the others. The per-rally
directory follows the TT3D contract documented in docs/DATASET_SCHEMA.md.
"""
from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import ball, body, config, pose2d, reorient, table, video
from .procutil import LOG, ffprobe

ALL_STAGES = ("canonical", "table", "body", "ball")


@dataclass
class RallyResult:
    rally_dir: str
    ok: bool = False
    n_frames: int = 0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    stages: dict = field(default_factory=dict)      # stage -> "ok" | "error: ..."
    ball_coverage: float = 0.0
    has_ball_2d: bool = False
    has_ball_3d: bool = False
    has_pose_p0: bool = False
    has_pose_p1: bool = False
    calib_ok: bool = False


def _ball_coverage(rally_dir: Path) -> float:
    csv = rally_dir / "ball_traj_2D.csv"
    if not csv.exists():
        return 0.0
    try:
        df = pd.read_csv(csv)
        if "Visibility" in df and len(df):
            return float((df["Visibility"].astype(int) != 0).mean())
    except Exception:
        pass
    return 0.0


def process_rally(
    src_clip: Path | str,
    rally_dir: Path | str,
    cfg: config.PipelineConfig,
    meta_extra: dict | None = None,
    stages: tuple[str, ...] = ALL_STAGES,
    force: bool = False,
) -> RallyResult:
    src_clip = Path(src_clip)
    rally_dir = Path(rally_dir).resolve()
    rally_dir.mkdir(parents=True, exist_ok=True)
    res = RallyResult(rally_dir=str(rally_dir))
    rally_mp4 = rally_dir / "rally.mp4"

    # --- canonical video ---------------------------------------------------
    try:
        info = video.make_canonical_rally(src_clip, rally_mp4, fps=cfg.canonical_fps)
        res.n_frames, res.fps = info["n_frames"], info["fps"]
        res.width, res.height = info["width"], info["height"]
        res.stages["canonical"] = "ok"
    except Exception as exc:  # cannot proceed without a clip
        res.stages["canonical"] = f"error: {exc}"
        _write_meta(rally_dir, res, meta_extra)
        return res

    dur = res.n_frames / res.fps if res.fps else 0.0
    if dur < cfg.min_rally_seconds:
        res.stages["skip"] = f"too short ({dur:.1f}s)"
        _write_meta(rally_dir, res, meta_extra)
        return res

    # A calibration may already exist from a previous (partial) run — honour it
    # so body/ball can run without re-selecting the table stage.
    res.calib_ok = (rally_dir / "camera.yaml").exists()

    # --- table (camera calibration) ---------------------------------------
    if "table" in stages:
        try:
            table.run_table(rally_dir, res.width, res.height, cfg, force=force)
            res.calib_ok = (rally_dir / "camera.yaml").exists()
            res.stages["table"] = "ok"
        except Exception as exc:
            res.stages["table"] = f"error: {exc}"
            LOG.warning("[table] %s failed: %s", rally_dir.name, exc)

    # --- body (2D pose -> 3D -> world) -------------------------------------
    if "body" in stages and res.calib_ok:
        try:
            pose2d.generate_mb_input(rally_dir, res.n_frames, cfg, force=force)
            # Resolve the 90-deg table corner-assignment ambiguity before any
            # stage consumes camera.yaml (alignment + ball 3D depend on it).
            reorient.check_and_fix(rally_dir)
            body.run_body(rally_dir, cfg, force=force)
            res.has_pose_p0 = (rally_dir / "p0_3d.npy").exists()
            res.has_pose_p1 = (rally_dir / "p1_3d.npy").exists()
            res.stages["body"] = "ok"
        except Exception as exc:
            res.stages["body"] = f"error: {exc}"
            LOG.warning("[body] %s failed: %s", rally_dir.name, exc)
    elif "body" in stages:
        res.stages["body"] = "skipped: no calibration"

    # --- ball (2D detect -> 3D reconstruct) --------------------------------
    if "ball" in stages and res.calib_ok:
        try:
            ball.run_ball(rally_dir, cfg, force=force)
            res.has_ball_2d = (rally_dir / "ball_traj_2D.csv").exists()
            res.has_ball_3d = (rally_dir / "ball_traj_3D.csv").exists()
            res.stages["ball"] = "ok" if res.has_ball_3d else "ok (2D only)"
        except Exception as exc:
            res.has_ball_2d = (rally_dir / "ball_traj_2D.csv").exists()
            res.stages["ball"] = f"error: {exc}"
            LOG.warning("[ball] %s failed: %s", rally_dir.name, exc)
    elif "ball" in stages:
        res.stages["ball"] = "skipped: no calibration"

    # Finalize flags from disk so meta is accurate even when only a subset of
    # stages ran this invocation (e.g. re-running just the ball stage).
    res.calib_ok = res.calib_ok or (rally_dir / "camera.yaml").exists()
    res.has_pose_p0 = res.has_pose_p0 or (rally_dir / "p0_3d.npy").exists()
    res.has_pose_p1 = res.has_pose_p1 or (rally_dir / "p1_3d.npy").exists()
    res.has_ball_2d = res.has_ball_2d or (rally_dir / "ball_traj_2D.csv").exists()
    res.has_ball_3d = res.has_ball_3d or (rally_dir / "ball_traj_3D.csv").exists()

    res.ball_coverage = _ball_coverage(rally_dir)
    res.ok = res.calib_ok and (
        res.has_ball_2d or res.has_ball_3d or (res.has_pose_p0 and res.has_pose_p1)
    )
    _write_meta(rally_dir, res, meta_extra)
    return res


def _write_meta(rally_dir: Path, res: RallyResult, meta_extra: dict | None) -> None:
    meta = asdict(res)
    meta["quality"] = {
        "calib_ok": res.calib_ok,
        "ball_coverage": round(res.ball_coverage, 3),
        "has_ball_2d": res.has_ball_2d,
        "has_ball_3d": res.has_ball_3d,
        "has_pose_p0": res.has_pose_p0,
        "has_pose_p1": res.has_pose_p1,
    }
    if meta_extra:
        meta.update(meta_extra)
    (rally_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
