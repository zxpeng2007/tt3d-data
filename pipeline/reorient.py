"""Camera orientation disambiguation.

The table calibration solves camera pose + focal from the table's four corners
and its known dimensions (2.74 x 1.525 m). For a rectangle this correspondence
has a discrete ambiguity: assigning image corners to world corners rotated by
90 deg also fits the image quad almost perfectly (the projected outline still
lands on the table), but the world frame ends up rotated 90 deg -- length and
width swapped. Everything ON the table plane still projects fine, while
everything OFF the plane (ball flight, player positions) is stretched along the
camera's depth axis and mis-oriented relative to the table.

Disambiguation uses physical priors that hold for every table-tennis rally:
  * players stand behind OPPOSITE table ENDS (|y| > ~1.0 m, opposite signs),
    never off the sides;
  * ray-casting their ankles to the floor plane (z = -0.76, a known constant)
    gives their table-frame positions under a candidate camera.

`check_and_fix()` scores the current camera and the re-solved 90 deg
alternative, and rewrites camera.yaml (backing up the original) when the
alternative is clearly the physical one.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import config
from .procutil import LOG

_HW, _HL = config.TABLE_WIDTH / 2, config.TABLE_LENGTH / 2
# world corners, cyclic order (matches upstream table_calibrator convention)
CORNERS = np.array(
    [[-_HW, -_HL, 0], [_HW, -_HL, 0], [_HW, _HL, 0], [-_HW, _HL, 0]], dtype=np.float64
)
FLOOR_Z = -config.TABLE_HEIGHT          # floor plane in the table frame
ANKLES = (15, 16)                       # COCO/Halpe ankle joint indices


def _K(f: float, w: int, h: int) -> np.ndarray:
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def _project(pts, rvec, tvec, K):
    img, _ = cv2.projectPoints(
        np.asarray(pts, np.float64).reshape(-1, 1, 3),
        np.asarray(rvec, np.float64).reshape(3, 1),
        np.asarray(tvec, np.float64).reshape(3, 1), K, None)
    return img.reshape(-1, 2)


def _solve_assignment(quad: np.ndarray, world: np.ndarray, w: int, h: int):
    """Solve pose + focal for a given image-quad <-> world-corner assignment.

    Grid-initialize f with planar PnP, then jointly refine (rvec, tvec, f) by
    minimizing corner reprojection error. Returns (rvec, tvec, f, rmse_px).
    """
    from scipy.optimize import minimize

    obj = world.reshape(-1, 1, 3).astype(np.float64)
    img = quad.reshape(-1, 1, 2).astype(np.float64)
    best = None
    for f0 in (600, 1000, 1600, 2400, 3400, 5000, 7000):
        K0 = _K(f0, w, h)
        for flag in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE):
            try:
                ok, rv, tv = cv2.solvePnP(obj, img, K0, None, flags=flag)
            except cv2.error:
                continue
            if not ok or float(tv[2]) <= 0:
                continue
            err = float(np.sqrt(np.mean(
                np.sum((_project(world, rv, tv, K0) - quad) ** 2, axis=1))))
            if best is None or err < best[3]:
                best = (rv.ravel(), tv.ravel(), float(f0), err)
    if best is None:
        return None

    def cost(p):
        rv, tv, f = p[:3], p[3:6], p[6]
        if f < 300 or tv[2] <= 0:
            return 1e9
        d = _project(world, rv, tv, _K(f, w, h)) - quad
        return float(np.mean(np.sum(d * d, axis=1)))

    x0 = np.concatenate([best[0], best[1], [best[2]]])
    res = minimize(cost, x0, method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
    rv, tv, f = res.x[:3], res.x[3:6], float(res.x[6])
    rmse = float(np.sqrt(cost(res.x)))
    return rv, tv, f, rmse


def _floor_positions(mb_input: Path, rvec, tvec, K, score_thr=0.5):
    """Median table-frame floor position per player (ankles ray-cast to z=FLOOR_Z)."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64))
    t = np.asarray(tvec, np.float64).ravel()
    Kinv = np.linalg.inv(K)
    n = R @ np.array([0, 0, 1.0])
    p0 = R @ np.array([0, 0, FLOOR_Z]) + t
    acc = {0: [], 1: []}
    for it in json.loads(mb_input.read_text(encoding="utf-8")):
        kp = np.array(it["keypoints"]).reshape(-1, 3)
        for j in ANKLES:
            if kp[j, 2] > score_thr:
                d = Kinv @ np.array([kp[j, 0], kp[j, 1], 1.0])
                denom = n @ d
                if abs(denom) < 1e-9:
                    continue
                s = (n @ p0) / denom
                if s <= 0:
                    continue
                acc[it["idx"]].append(R.T @ (s * d - t))
    out = {}
    for pid, pts in acc.items():
        if pts:
            out[pid] = np.median(np.array(pts), axis=0)
    return out


def _penalty(pos: dict) -> float:
    """0 = physically ideal (players behind opposite ends, inside side lines)."""
    if len(pos) < 2:
        return 10.0
    (x0, y0, _), (x1, y1, _) = pos[0], pos[1]
    pen = 0.0
    if np.sign(y0) == np.sign(y1):
        pen += 4.0                                   # same end: impossible
    pen += max(0.0, abs(x0) - 1.3) + max(0.0, abs(x1) - 1.3)   # off the sides
    pen += max(0.0, 1.0 - abs(y0)) + max(0.0, 1.0 - abs(y1))   # not behind ends
    return float(pen)


def check_and_fix(rally_dir: Path | str, margin: float = 1.0) -> bool:
    """Validate camera orientation for a rally; fix camera.yaml if 90-deg swapped.

    Requires camera.yaml and mb_input.json. Returns True if the camera was
    rewritten. Non-fatal: any failure leaves the camera untouched.
    """
    rally_dir = Path(rally_dir)
    cam_path = rally_dir / "camera.yaml"
    mb_input = rally_dir / "mb_input.json"
    if not cam_path.exists() or not mb_input.exists():
        return False
    try:
        cam = yaml.safe_load(cam_path.read_text(encoding="utf-8"))
        rvec = np.array(cam["rvec"], np.float64)
        tvec = np.array(cam["tvec"], np.float64)
        f, h, w = float(cam["f"]), int(cam["h"]), int(cam["w"])
        K = _K(f, w, h)

        # Observed image quad = current solution's projected corners (they sit
        # on the real table regardless of which assignment was solved).
        quad = _project(CORNERS, rvec, tvec, K)

        pen_cur = _penalty(_floor_positions(mb_input, rvec, tvec, K))
        # 90-deg alternative: cyclic-shift the world corners by one.
        alt = _solve_assignment(quad, CORNERS[[1, 2, 3, 0]], w, h)
        if alt is None:
            return False
        rv_a, tv_a, f_a, rmse_a = alt
        if rmse_a > 5.0:            # alternative must still fit the table corners
            return False
        pen_alt = _penalty(_floor_positions(mb_input, rv_a, tv_a, _K(f_a, w, h)))
        LOG.info("[reorient] %s penalty current=%.2f alt=%.2f (alt rmse %.2fpx, f %.0f)",
                 rally_dir.name, pen_cur, pen_alt, rmse_a, f_a)
        if pen_alt + margin >= pen_cur:
            return False            # current camera already the physical one

        backup = rally_dir / "camera_orig.yaml"
        if not backup.exists():
            shutil.copyfile(cam_path, backup)
        cam_out = {"rvec": [float(v) for v in rv_a],
                   "tvec": [float(v) for v in tv_a],
                   "f": int(round(f_a)), "h": int(h), "w": int(w)}
        cam_path.write_text(yaml.safe_dump(cam_out), encoding="utf-8")
        LOG.info("[reorient] %s camera REORIENTED (90-deg corner-assignment fix)",
                 rally_dir.name)
        return True
    except Exception as exc:            # never break the pipeline on QC
        LOG.warning("[reorient] %s check failed: %s", rally_dir.name, exc)
        return False
