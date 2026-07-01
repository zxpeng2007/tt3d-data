"""Body stage: MotionBERT 3D lifting + world-frame alignment.

1. Run pipeline/mb_infer.py inside the MotionBERT repo -> player_0.npy / player_1.npy
   (camera-frame 3D, shape (n_frames,17,3)).
2. Align each player into the table/world frame by solving a per-foot-contact
   translation + scale that minimizes 2D reprojection error, then interpolating and
   smoothing across frames. Self-contained port of upstream tt3d/pose/align.py
   (no blocking matplotlib, fps configurable) -> p0_3d.npy / p1_3d.npy.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import config
from .procutil import LOG, StageError, agg_env, run

_HALPE2H36M_SRC = [19, 12, 14, 16, 11, 13, 15, None, 18, 0, 17, 5, 7, 9, 6, 8, 10]


def _halpe2h36m(x: np.ndarray) -> np.ndarray:
    """(T,26,C) Halpe -> (T,17,C) H36M, matching MotionBERT's mapping."""
    T, _, C = x.shape
    y = np.zeros((T, 17, C), dtype=x.dtype)
    for dst, src in enumerate(_HALPE2H36M_SRC):
        if src is None:  # joint 7 (spine) = mid(neck18, hip19)
            y[:, 7] = (x[:, 18] + x[:, 19]) * 0.5
        else:
            y[:, dst] = x[:, src]
    return y


def _get_K(f, h, w):
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def _get_transform(rvec, tvec):
    import cv2
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec).reshape(3)
    return T


def _low_vel_acc_indices(pos, dt, threshold=0.1):
    vel = np.diff(pos) / dt
    acc = np.diff(vel) / dt
    low_v = np.where(np.abs(vel) < threshold)[0]
    low_a = np.where(np.abs(acc) < threshold / dt)[0]
    return np.intersect1d(low_v, low_a)


def _reprojection_error(params, P_3d, P_2d, K, T_table):
    t = np.array(params[:3])
    scale = params[3]
    P_table = scale * (P_3d + t)
    P_cam = (P_table @ T_table[:3, :3].T) + T_table[:3, 3]
    proj = (K @ P_cam.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    err = np.linalg.norm(P_2d - proj, axis=1)
    reg = np.sum(np.abs(P_table[6::17, 2] + 0.66)) + np.sum(np.abs(P_table[3::17, 2] + 0.66))
    return np.mean(err) + 10 * reg


def _solve_Tt(P_2d, P_3d, K, T_table):
    from scipy.optimize import minimize
    x0 = np.zeros(4)
    x0[3] = 1.0
    res = minimize(_reprojection_error, x0, args=(P_3d, P_2d, K, T_table), method="BFGS")
    return res.x[:3], res.x[3]


def _smooth(tvecs, window=5):
    tvecs = np.asarray(tvecs)
    pad = window // 2
    padded = np.pad(tvecs, ((pad, pad), (0, 0)), mode="edge")
    return np.stack(
        [np.convolve(padded[:, i], np.ones(window) / window, mode="valid") for i in range(3)],
        axis=1,
    )


def _align_player(pose_cam, pose_2d, K, T_table, fps):
    """pose_cam:(N,17,3) camera frame; pose_2d:(N,17,2). Returns (N,17,3) world frame."""
    from scipy.interpolate import interp1d
    T_inv = np.linalg.inv(T_table)
    pose_rot = pose_cam @ T_inv[:3, :3].T  # orientation only
    n = min(len(pose_rot), len(pose_2d))
    pose_rot, pose_2d = pose_rot[:n], pose_2d[:n]
    dt = 1.0 / fps

    # foot-contact frames: ankles (idx 3, 6) with low vertical velocity/accel
    idx_r = _low_vel_acc_indices(pose_rot[:, 3, 2], dt, 0.1)
    idx_l = _low_vel_acc_indices(pose_rot[:, 6, 2], dt, 0.1)
    contact = np.unique(np.concatenate([idx_r, idx_l]))
    if len(contact) < 2:
        contact = np.arange(n)

    Ts, Ss = [], []
    for i in contact:
        t, s = _solve_Tt(pose_2d[i], pose_rot[i], K, T_table)
        Ts.append(t)
        Ss.append(s)
    Ts = np.array(Ts)
    s = float(np.mean(Ss))
    interp = interp1d(contact, Ts, axis=0, kind="linear", fill_value="extrapolate")
    tvec = _smooth(interp(np.arange(n)))
    return np.stack([s * (pose_rot[i] + tvec[i]) for i in range(n)]).astype(np.float32)


def run_body(rally_dir: Path | str, cfg: config.PipelineConfig, force: bool = False) -> tuple[Path, Path]:
    rally_dir = Path(rally_dir).resolve()
    p0_out, p1_out = rally_dir / "p0_3d.npy", rally_dir / "p1_3d.npy"
    if p0_out.exists() and p1_out.exists() and not force:
        LOG.info("[body] p0/p1_3d.npy exist, skipping (%s)", rally_dir.name)
        return p0_out, p1_out

    mb_input = rally_dir / "mb_input.json"
    camera_yaml = rally_dir / "camera.yaml"
    for req in (mb_input, camera_yaml):
        if not req.exists():
            raise StageError(f"[body] missing prerequisite {req.name} in {rally_dir}")
    if not config.MOTIONBERT_CKPT.exists():
        raise StageError(f"MotionBERT checkpoint missing: {config.MOTIONBERT_CKPT}")

    # 1) MotionBERT inference (subprocess inside the MotionBERT repo)
    env = agg_env()
    prev = os.environ.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(config.MOTIONBERT_DIR)] + ([prev] if prev else []))
    run(
        [config.PYTHON, str((Path(__file__).parent / "mb_infer.py").resolve()),
         "--config", "configs/pose3d/MB_ft_h36m_global_lite.yaml",
         "--ckpt", str(config.MOTIONBERT_CKPT),
         "--json", str(mb_input),
         "--out_dir", str(rally_dir),
         "--players", "0", "1"],
        cwd=config.MOTIONBERT_DIR, env=env,
        log_path=config.LOGS_DIR / f"body_{rally_dir.name}.log",
    )
    player0 = rally_dir / "player_0.npy"
    player1 = rally_dir / "player_1.npy"
    if not player0.exists() or not player1.exists():
        raise StageError("[body] MotionBERT did not produce player_0/1.npy")

    # 2) Alignment (in-process, self-contained)
    import yaml
    cam = yaml.safe_load(camera_yaml.read_text(encoding="utf-8"))
    rvec, tvec, f = cam["rvec"], cam["tvec"], cam["f"]
    h, w = cam["h"], cam["w"]
    K = _get_K(f, h, w)
    T_table = _get_transform(rvec, tvec)

    instances = json.loads(mb_input.read_text(encoding="utf-8"))

    def _pose2d_for(idx):
        kk = [np.array(it["keypoints"]).reshape(-1, 3)[:, :2]
              for it in instances if it["idx"] == idx]
        arr = np.array(kk)[None] if kk and kk[0].ndim == 1 else np.array(kk)
        return _halpe2h36m(arr.reshape(len(kk), -1, 2))

    pose2d_0 = _pose2d_for(0)
    pose2d_1 = _pose2d_for(1)
    pose_cam_0 = np.load(player0)
    pose_cam_1 = np.load(player1)

    fps = cfg.canonical_fps
    p0_world = _align_player(pose_cam_0, pose2d_0, K, T_table, fps)
    p1_world = _align_player(pose_cam_1, pose2d_1, K, T_table, fps)
    np.save(p0_out, p0_world)
    np.save(p1_out, p1_world)
    LOG.info("[body] aligned %s: p0=%s p1=%s", rally_dir.name, p0_world.shape, p1_world.shape)
    return p0_out, p1_out
