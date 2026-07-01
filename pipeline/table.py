"""Table stage: camera calibration + table position.

Wraps upstream TT3D calibration scripts. They use bare imports (``from
table_calibrator import ...``) and a cwd-relative model path
(``./weights/table_segmentation.ckpt``), so we invoke them with:

    cwd      = <tt3d repo root>              # so ./weights/... resolves
    PYTHONPATH += <tt3d repo>/tt3d/calibration  # so bare imports resolve

Outputs, written into the rally dir:
    cam_cal.csv   raw per-frame (rvec, tvec, f) observations
    camera.yaml   Kalman-filtered single static-camera calibration
    table.json    table corner world coords (the world/origin frame)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import config
from .procutil import LOG, StageError, agg_env, run

CALIB_DIR = config.TT3D_DIR / "tt3d" / "calibration"


def _calib_env() -> dict:
    env = agg_env()
    prev = os.environ.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(CALIB_DIR)] + ([prev] if prev else []))
    return env


def _write_table_json(rally_dir: Path) -> None:
    """Record the regulation table geometry (world frame origin) for reference."""
    hl, hw = config.TABLE_LENGTH / 2.0, config.TABLE_WIDTH / 2.0
    corners = {
        "convention": "table centre = origin; long axis = Y; surface z = 0 (m)",
        "length_m": config.TABLE_LENGTH,
        "width_m": config.TABLE_WIDTH,
        "net_height_m": config.NET_HEIGHT,
        "surface_height_m": config.TABLE_HEIGHT,
        "corners_world": [
            [-hw, -hl, 0.0], [hw, -hl, 0.0], [hw, hl, 0.0], [-hw, hl, 0.0],
        ],
    }
    (rally_dir / "table.json").write_text(json.dumps(corners, indent=2), encoding="utf-8")


def run_table(
    rally_dir: Path | str,
    width: int,
    height: int,
    cfg: config.PipelineConfig,
    force: bool = False,
) -> Path:
    """Calibrate the camera for a rally. Returns the camera.yaml path.

    Assumes rally_dir contains a canonical ``rally.mp4``. Idempotent unless force.
    """
    rally_dir = Path(rally_dir).resolve()
    rally_mp4 = rally_dir / "rally.mp4"
    cam_cal = rally_dir / "cam_cal.csv"
    camera_yaml = rally_dir / "camera.yaml"

    if not rally_mp4.exists():
        raise StageError(f"rally.mp4 missing in {rally_dir}")

    if camera_yaml.exists() and not force:
        LOG.info("[table] camera.yaml exists, skipping (%s)", rally_dir.name)
        _write_table_json(rally_dir)
        return camera_yaml

    if not config.TABLE_SEG_CKPT.exists():
        raise StageError(
            f"Table segmentation weight missing: {config.TABLE_SEG_CKPT} "
            "(it ships with upstream TT3D; run scripts/download_weights.py)"
        )

    env = _calib_env()
    log = config.LOGS_DIR / f"table_{rally_dir.name}.log"

    # 1) Raw per-frame observations
    run(
        [config.PYTHON, "tt3d/calibration/calibrate.py", str(rally_mp4), "-o", str(cam_cal)],
        cwd=config.TT3D_DIR, env=env, log_path=log,
    )
    if not cam_cal.exists():
        raise StageError(f"calibrate.py did not produce {cam_cal}")

    # 2) Kalman-filter to a single static camera -> camera.yaml (next to cam_cal.csv)
    run(
        [config.PYTHON, "tt3d/calibration/filter.py", str(cam_cal),
         "--so", "-w", str(width), "-he", str(height)],
        cwd=config.TT3D_DIR, env=env, log_path=log, check=True,
    )
    if not camera_yaml.exists():
        raise StageError(f"filter.py did not produce {camera_yaml}")

    _write_table_json(rally_dir)
    LOG.info("[table] calibrated %s", rally_dir.name)
    return camera_yaml
