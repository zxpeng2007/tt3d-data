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
         f"tracker.max_disp={cfg.blurball_max_disp}",
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
    _bridge_gaps(out_csv, cfg)
    LOG.info("[ball] 2D detections -> %s", out_csv.name)
    return out_csv


def _bridge_gaps(csv_path: Path, cfg: config.PipelineConfig) -> None:
    """Fill short detection gaps with a local quadratic fit (smash recovery).

    Fast balls blur into weak detections and drop out for a few frames right
    when the trajectory is most interesting. Between two visible stretches the
    2D path is smooth (projectile + projection), so a quadratic fit through
    the nearest neighbours reconstructs the missing frames well. Filled rows
    get Visibility=1 (so downstream TT3D uses them) and Interp=1 (so training
    code can distinguish real detections from bridged ones).
    """
    import numpy as np
    import pandas as pd

    max_gap = int(getattr(cfg, "ball_bridge_max_gap", 0))
    if max_gap <= 0:
        return
    df = pd.read_csv(csv_path)
    # BlurBall writes X/Y as ints; the fitted fills are floats
    for col in ("X", "Y", "L", "Theta"):
        df[col] = df[col].astype(float)
    if "Interp" not in df.columns:
        df["Interp"] = 0
    vis = df[df.Visibility != 0]
    if len(vis) < 6:
        return
    vis_frames = vis.Frame.to_numpy()
    n_filled = 0
    for i in range(len(vis_frames) - 1):
        f0, f1 = int(vis_frames[i]), int(vis_frames[i + 1])
        gap = f1 - f0 - 1
        if gap < 1 or gap > max_gap:
            continue
        # nearest visible neighbours on each side (up to 3 per side)
        left = vis[vis.Frame <= f0].tail(3)
        right = vis[vis.Frame >= f1].head(3)
        support = pd.concat([left, right])
        if len(support) < 4:
            continue
        t = support.Frame.to_numpy(dtype=float)
        deg = 2 if len(support) >= 5 else 1
        px = np.polyfit(t, support.X.to_numpy(dtype=float), deg)
        py = np.polyfit(t, support.Y.to_numpy(dtype=float), deg)
        for f in range(f0 + 1, f1):
            x, y = float(np.polyval(px, f)), float(np.polyval(py, f))
            m = df.Frame == f
            if not m.any():
                continue
            df.loc[m, ["X", "Y", "Visibility", "Interp"]] = [x, y, 1, 1]
            df.loc[m, ["L", "Theta"]] = [0.0, 0.0]
            n_filled += 1
    if n_filled:
        df.to_csv(csv_path, index=False)
        LOG.info("[ball] bridged %d frames across short gaps", n_filled)


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
    _filter_ball_3d(out_csv, cfg)
    return out_csv


def _filter_ball_3d(csv_path: Path, cfg: config.PipelineConfig) -> None:
    """Drop physically-impossible reconstructed 3D ball points.

    rally.py's per-segment physics solve occasionally diverges, placing points
    below the table or far outside the play volume. The table is a fixed metric
    reference, so we can reject anything on the wrong side of it.
    """
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return
    hw, hl = config.TABLE_WIDTH / 2, config.TABLE_LENGTH / 2
    on_surface = df["z"].abs() < 0.06
    keep = ((df["z"] > -0.12) & (df["z"] < 1.5)
            & (df["x"].abs() < hw + 0.9) & (df["y"].abs() < hl + 0.8)
            # a bounce (near surface) must actually be on the table
            & ~(on_surface & ((df["x"].abs() > hw + 0.15) | (df["y"].abs() > hl + 0.15))))
    dropped = int((~keep).sum())
    if dropped:
        df[keep].to_csv(csv_path, index=False)
        LOG.info("[ball] dropped %d implausible 3D points (kept %d)", dropped, int(keep.sum()))


def run_ball(rally_dir: Path | str, cfg: config.PipelineConfig, force: bool = False) -> Path:
    """Produce 2D ball detections (required) and best-effort 3D reconstruction.

    The 2D track (ball_traj_2D.csv) is the primary ball-position deliverable and is
    always kept. The 3D physics solve (rally.py) is fragile on clips that aren't a
    single clean rally, so its failure is non-fatal: we log and return the 2D path.
    """
    rally_dir = Path(rally_dir).resolve()
    out_csv = rally_dir / "ball_traj_3D.csv"
    if out_csv.exists() and not force:
        LOG.info("[ball] ball_traj_3D.csv exists, skipping (%s)", rally_dir.name)
        return out_csv
    ball2d = _blurball_2d(rally_dir, cfg)   # raises if BlurBall itself fails
    try:
        return _reconstruct_3d(rally_dir, cfg)
    except StageError as exc:
        LOG.warning("[ball] 3D reconstruction failed for %s (keeping 2D): %s",
                    rally_dir.name, str(exc).splitlines()[0])
        return ball2d
