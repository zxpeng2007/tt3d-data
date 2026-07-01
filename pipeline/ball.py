"""Ball stage: BlurBall 2D detection -> TT3D 3D trajectory reconstruction.

1. Run BlurBall inference on the canonical rally.mp4. With filter=False it keeps
   every frame, so ball Frame indices stay 1:1 with our (already de-duplicated)
   canonical video. It writes <rally_dir>/frames_rally/traj.csv with columns
   Frame,X,Y,Visibility,L,Theta -- exactly TT3D's ball_traj_2D.csv schema, so we
   just move it into place.
2. Run TT3D rally.py to reconstruct the 3D trajectory (ray-plane bounce + casadi
   spin/velocity optimization) -> <rally_dir>/ball_traj_3D.csv.

rally.py calls plt.show() and assumes fps=25, so we run it with MPLBACKEND=Agg
(makes show() a no-op) on canonical 25fps clips. It uses tt3d package imports, so
we add the package + submodule dirs to PYTHONPATH.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from . import config
from .procutil import LOG, StageError, agg_env, run

RALLY_DIR = config.TT3D_DIR / "tt3d" / "rally"
TRAJSEG_DIR = config.TT3D_DIR / "tt3d" / "traj_seg"
CALIB_DIR = config.TT3D_DIR / "tt3d" / "calibration"


def _blurball_2d(rally_dir: Path, cfg: config.PipelineConfig) -> Path:
    rally_mp4 = rally_dir / "rally.mp4"
    out_csv = rally_dir / "ball_traj_2D.csv"
    if out_csv.exists():
        return out_csv
    if not config.BLURBALL_CKPT.exists():
        raise StageError(
            f"BlurBall weight missing: {config.BLURBALL_CKPT} (run scripts/download_weights.py)"
        )
    wasb_root = str(config.BLURBALL_DIR).replace("\\", "/")
    run_dir = str((rally_dir / "_bb").resolve()).replace("\\", "/")
    env = agg_env()
    env["WASB_ROOT"] = wasb_root
    run(
        [config.PYTHON, "src/main.py", "--config-name=inference_blurball",
         f"detector.model_path={str(config.BLURBALL_CKPT).replace(chr(92), '/')}",
         f"+input_vid={str(rally_mp4).replace(chr(92), '/')}",
         f"detector.step={cfg.blurball_step}",
         f"detector.postprocessor.score_threshold={cfg.blurball_score_threshold}",
         # Disable BlurBall's per-frame visualization (buggy draw_frame path); we
         # only need traj.csv, which is written regardless.
         "runner.vis_result=False", "runner.vis_hm=False", "runner.vis_traj=False",
         f"WASB_ROOT={wasb_root}",
         f"hydra.run.dir={run_dir}"],
        cwd=config.BLURBALL_DIR, env=env,
        log_path=config.LOGS_DIR / f"ball_{rally_dir.name}.log",
    )
    # BlurBall writes frames_<stem>/traj.csv next to the input video.
    traj = rally_dir / "frames_rally" / "traj.csv"
    if not traj.exists():
        # fall back: search for any traj.csv it produced
        hits = list(rally_dir.glob("**/traj.csv"))
        if not hits:
            raise StageError(f"BlurBall produced no traj.csv under {rally_dir}")
        traj = hits[0]
    shutil.copyfile(traj, out_csv)
    LOG.info("[ball] 2D detections -> %s", out_csv.name)
    return out_csv


def _reconstruct_3d(rally_dir: Path, cfg: config.PipelineConfig) -> Path:
    out_csv = rally_dir / "ball_traj_3D.csv"
    if not (rally_dir / "camera.yaml").exists():
        raise StageError("[ball] camera.yaml required for 3D reconstruction (run table stage)")
    env = agg_env()
    env["TT3D_FPS"] = str(cfg.canonical_fps)
    prev = os.environ.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(config.TT3D_DIR), str(RALLY_DIR), str(TRAJSEG_DIR), str(CALIB_DIR)]
        + ([prev] if prev else [])
    )
    run(
        [config.PYTHON, "tt3d/rally/rally.py", str(rally_dir)],
        cwd=config.TT3D_DIR, env=env,
        log_path=config.LOGS_DIR / f"ball3d_{rally_dir.name}.log",
    )
    if not out_csv.exists():
        raise StageError(f"rally.py did not produce {out_csv}")
    return out_csv


def run_ball(rally_dir: Path | str, cfg: config.PipelineConfig, force: bool = False) -> Path:
    rally_dir = Path(rally_dir).resolve()
    out_csv = rally_dir / "ball_traj_3D.csv"
    if out_csv.exists() and not force:
        LOG.info("[ball] ball_traj_3D.csv exists, skipping (%s)", rally_dir.name)
        return out_csv
    _blurball_2d(rally_dir, cfg)
    return _reconstruct_3d(rally_dir, cfg)
