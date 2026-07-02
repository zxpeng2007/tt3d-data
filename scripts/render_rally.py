"""Render a reconstructed rally into human-viewable visuals.

Produces, under <rally_dir>/renders/:
  overlay_2d.mp4  : ball detections + both players' 2D skeletons on the real video
  scene_3d.mp4    : 3D table + player skeletons + ball trajectory, animated
  summary.png     : static multi-view figure (3D, top-down, side, ball height)

  python scripts/render_rally.py --rally-dir data/dataset/rallies/<match>/<rally>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from pipeline import config

# The regulation table is a CONSTANT metric reference: projecting it with the
# estimated camera is both a calibration QC and the table outline drawn in 2D.
_W, _L, _NET = config.TABLE_WIDTH / 2, config.TABLE_LENGTH / 2, config.NET_HEIGHT
TABLE_CORNERS = np.array([[-_W, -_L, 0], [_W, -_L, 0], [_W, _L, 0], [-_W, _L, 0]], float)
NET_QUAD = np.array([[-_W, 0, 0], [_W, 0, 0], [_W, 0, _NET], [-_W, 0, _NET]], float)
CENTER_LINE = np.array([[0, -_L, 0], [0, _L, 0]], float)


def _load_cam(rally_dir: Path):
    c = yaml.safe_load((rally_dir / "camera.yaml").read_text(encoding="utf-8"))
    rvec = np.array(c["rvec"], float).reshape(3, 1)
    tvec = np.array(c["tvec"], float).reshape(3, 1)
    K = np.array([[c["f"], 0, c["w"] / 2], [0, c["f"], c["h"] / 2], [0, 0, 1]], float)
    return rvec, tvec, K


def _project(pts_world, rvec, tvec, K):
    img, _ = cv2.projectPoints(np.asarray(pts_world, float).reshape(-1, 1, 3), rvec, tvec, K, None)
    return img.reshape(-1, 2)


def _plausible_ball(df: pd.DataFrame) -> pd.DataFrame:
    """Drop physically-impossible reconstructed points using the table geometry.

    A near-surface point is a bounce and MUST lie on the table; higher points are
    flight and may extend modestly beyond the ends (players hit from behind them).
    The depth axis (Y) is the camera's line of sight and least-constrained, so this
    is what removes the divergent 'bounces past the end' that stretch the plot.
    """
    on_surface = df.z.abs() < 0.06
    keep = ((df.z > -0.12) & (df.z < 1.2) & (df.x.abs() < 1.1) & (df.y.abs() < 2.2)
            & ~(on_surface & ((df.x.abs() > _W + 0.15) | (df.y.abs() > _L + 0.15))))
    return df[keep]

# H36M-17 skeleton (MotionBERT output order)
H36M_EDGES = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8),
              (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]
# COCO-17 skeleton (rtmlib 2D order; == Halpe 0..16)
COCO_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
              (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
BLUE = (255, 140, 0)   # player 0 (BGR)
RED = (60, 60, 255)    # player 1 (BGR)
GREEN = (0, 230, 0)


def _ball_by_frame(rally_dir: Path, n_frames: int):
    df = pd.read_csv(rally_dir / "ball_traj_3D.csv").drop_duplicates("idx")
    df = _plausible_ball(df)
    pos = {int(r.idx): (r.x, r.y, r.z) for r in df.itertuples()}
    xs = np.array(sorted(pos))
    return pos, xs, df


def _gap_break(frames, pts, max_gap: int = 3):
    """Insert NaN rows where consecutive frames are more than max_gap apart so
    matplotlib breaks the line instead of drawing a straight glitch segment."""
    pts = np.asarray(pts, float)
    out = []
    for i, p in enumerate(pts):
        if i and frames[i] - frames[i - 1] > max_gap:
            out.append([np.nan] * pts.shape[1])
        out.append(p)
    return np.array(out)


# ----------------------------- 2D overlay -----------------------------------
def render_2d(rally_dir: Path, out: Path) -> None:
    cap = cv2.VideoCapture(str(rally_dir / "rally.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    b2 = pd.read_csv(rally_dir / "ball_traj_2D.csv")
    inst = json.loads((rally_dir / "mb_input.json").read_text(encoding="utf-8"))
    poses = {0: {}, 1: {}}
    for it in inst:
        f = int(it["image_id"].split(".")[0])
        poses[it["idx"]][f] = np.array(it["keypoints"]).reshape(-1, 3)

    # Project the constant-geometry table with the estimated camera (calibration QC).
    rvec, tvec, K = _load_cam(rally_dir)
    tbl = _project(TABLE_CORNERS, rvec, tvec, K).astype(int)
    net = _project(NET_QUAD, rvec, tvec, K).astype(int)
    ctr = _project(CENTER_LINE, rvec, tvec, K).astype(int)

    writer = imageio.get_writer(out, fps=fps, macro_block_size=None)
    trail = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # projected table outline (yellow), net (cyan), centre line
        cv2.polylines(frame, [tbl], True, (0, 255, 255), 2)
        cv2.polylines(frame, [net], True, (255, 220, 0), 2)
        cv2.line(frame, tuple(ctr[0]), tuple(ctr[1]), (0, 255, 255), 1)
        # 2D skeletons
        for pid, color in ((0, BLUE), (1, RED)):
            kp = poses[pid].get(i)
            if kp is None:
                continue
            for a, b in COCO_EDGES:
                if kp[a, 2] > 0.3 and kp[b, 2] > 0.3:
                    cv2.line(frame, tuple(kp[a, :2].astype(int)), tuple(kp[b, :2].astype(int)), color, 2)
            for j in range(17):
                if kp[j, 2] > 0.3:
                    cv2.circle(frame, tuple(kp[j, :2].astype(int)), 3, color, -1)
        # ball + trail
        row = b2[b2.Frame == i]
        if len(row) and int(row.Visibility.iloc[0]) != 0:
            x, y = int(row.X.iloc[0]), int(row.Y.iloc[0])
            trail.append((x, y))
        trail = trail[-15:]
        for k in range(1, len(trail)):
            cv2.line(frame, trail[k - 1], trail[k], GREEN, 2)
        if trail:
            cv2.circle(frame, trail[-1], 6, GREEN, -1)
            cv2.circle(frame, trail[-1], 9, (255, 255, 255), 2)
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    writer.close()


# ----------------------------- 3D helpers -----------------------------------
def _draw_table(ax):
    L, W, net = config.TABLE_LENGTH / 2, config.TABLE_WIDTH / 2, config.NET_HEIGHT
    # surface
    ax.plot([-W, W, W, -W, -W], [-L, -L, L, L, -L], [0, 0, 0, 0, 0], color="#1f6f4a", lw=2)
    ax.plot([-W, W], [0, 0], [0, 0], color="#1f6f4a", lw=1, alpha=0.6)  # centre line at net
    # net
    ax.plot([-W, W, W, -W, -W], [0, 0, 0, 0, 0], [0, 0, net, net, 0], color="#888", lw=1.5)


def _draw_skel(ax, pose, color):
    for a, b in H36M_EDGES:
        ax.plot(*[[pose[a, k], pose[b, k]] for k in range(3)], color=color, lw=2)


def _setup(ax, ball_xyz):
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_xlim(-2, 2); ax.set_ylim(-3, 3.5); ax.set_zlim(-1, 1.2)
    try:
        ax.set_box_aspect((4, 6.5, 2.2))
    except Exception:
        pass


def render_3d(rally_dir: Path, out: Path) -> None:
    p0 = np.load(rally_dir / "p0_3d.npy"); p1 = np.load(rally_dir / "p1_3d.npy")
    n = min(len(p0), len(p1))
    pos, xs, _ = _ball_by_frame(rally_dir, n)
    bt = np.array([pos[k] for k in xs])
    writer = imageio.get_writer(out, fps=20, macro_block_size=None)
    for i in range(n):
        fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")
        _draw_table(ax)
        _draw_skel(ax, p0[i], "#1f77ff"); _draw_skel(ax, p1[i], "#ff3b3b")
        seen = xs[xs <= i]
        if len(seen):
            tr = _gap_break(seen, [pos[k] for k in seen])
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color="#00c000", lw=1.5, alpha=0.7)
            last = np.array(pos[seen[-1]])
            ax.scatter(*last, color="#00c000", s=40)
        _setup(ax, bt); ax.view_init(elev=18, azim=-70)
        fig.tight_layout()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        writer.append_data(img)
        plt.close(fig)
    writer.close()


def render_summary(rally_dir: Path, out: Path) -> None:
    p0 = np.load(rally_dir / "p0_3d.npy"); p1 = np.load(rally_dir / "p1_3d.npy")
    n = min(len(p0), len(p1))
    pos, xs, df = _ball_by_frame(rally_dir, n)
    bt = _gap_break(xs, [pos[k] for k in xs])
    mid = xs[len(xs) // 2]

    fig = plt.figure(figsize=(15, 4.5))
    # 3D
    ax = fig.add_subplot(141, projection="3d")
    _draw_table(ax); _draw_skel(ax, p0[mid], "#1f77ff"); _draw_skel(ax, p1[mid], "#ff3b3b")
    ax.plot(bt[:, 0], bt[:, 1], bt[:, 2], color="#00c000", lw=2)
    _setup(ax, bt); ax.view_init(elev=20, azim=-70); ax.set_title("3D scene (frame %d)" % mid)
    # top-down: table LENGTH horizontal (conventional), net = vertical line at Y=0
    ax2 = fig.add_subplot(142)
    W, L = config.TABLE_WIDTH / 2, config.TABLE_LENGTH / 2
    ax2.add_patch(plt.Rectangle((-L, -W), 2 * L, 2 * W, fill=False, ec="#1f6f4a", lw=2))
    ax2.axvline(0, color="#1f6f4a", lw=1.5, alpha=0.7)  # net
    ax2.plot(bt[:, 1], bt[:, 0], color="#00c000", lw=2, label="ball")
    ax2.plot(p0[:, 0, 1], p0[:, 0, 0], ".", color="#1f77ff", ms=3, label="player0")
    ax2.plot(p1[:, 0, 1], p1[:, 0, 0], ".", color="#ff3b3b", ms=3, label="player1")
    ax2.set_title("Top-down (length = horizontal)")
    ax2.set_xlabel("Y  table length (m)"); ax2.set_ylabel("X  width (m)")
    ax2.set_aspect("equal"); ax2.legend(fontsize=7)
    # side (Y-Z): ball flight + bounce
    ax3 = fig.add_subplot(143)
    ax3.plot(bt[:, 1], bt[:, 2], "-o", color="#00c000", ms=3)
    ax3.axhline(0, color="#1f6f4a", lw=2)
    ax3.set_title("Ball flight (side Y-Z)"); ax3.set_xlabel("Y (m)"); ax3.set_ylabel("Z height (m)")
    # ball height vs frame
    ax4 = fig.add_subplot(144)
    xs_b = _gap_break(xs, np.asarray(xs, float).reshape(-1, 1))[:, 0]
    ax4.plot(xs_b, bt[:, 2], "-o", color="#00c000", ms=3)
    ax4.axhline(0, color="#1f6f4a", lw=1)
    ax4.set_title("Ball height vs frame"); ax4.set_xlabel("frame"); ax4.set_ylabel("Z (m)")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rally-dir", type=Path, required=True)
    ap.add_argument("--skip-2d", action="store_true")
    ap.add_argument("--skip-3d", action="store_true")
    args = ap.parse_args()
    d = args.rally_dir
    out = d / "renders"; out.mkdir(exist_ok=True)
    has_3d = (d / "ball_traj_3D.csv").exists()
    if has_3d:
        print("summary.png ..."); render_summary(d, out / "summary.png")
    else:
        print("no ball_traj_3D.csv — skipping summary/scene (2D overlay only)")
    if not args.skip_2d:
        print("overlay_2d.mp4 ..."); render_2d(d, out / "overlay_2d.mp4")
    if has_3d and not args.skip_3d:
        print("scene_3d.mp4 ..."); render_3d(d, out / "scene_3d.mp4")
    print("done ->", out)


if __name__ == "__main__":
    main()
