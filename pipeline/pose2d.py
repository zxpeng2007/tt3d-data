"""2D pose stage: rtmlib (RTMDet+RTMPose) -> tracked players -> MotionBERT input.

Replaces upstream TT3D's mmpose-based generate_2d_pose.py + cvt_json.py (which need
the heavy mmcv/mmdet stack) with rtmlib (ONNXRuntime, no mmcv). We:

  1. Run rtmlib Body per frame -> COCO-17 keypoints + scores for every person.
  2. Derive a bbox per person from its keypoints and IoU-track across frames.
  3. Select the two players (most persistent central tracks) and assign idx 0/1
     (idx 0 = far side / smaller mean-y, idx 1 = near side).
  4. Emit ONE dense entry per (frame, player) — gaps interpolated — so the frame
     index maps 1:1 to the canonical video (required by align.py / ball / camera).
  5. Expand COCO-17 -> Halpe-26 (Head/Neck/Hip synthesized) which is what
     MotionBERT's WildDetDataset (halpe2h36m) expects.

Output: <rally_dir>/mb_input.json  (AlphaPose-style list of per-frame instances).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config
from .procutil import LOG, StageError

# COCO-17 index reference (== Halpe 0..16):
# 0 nose,1 leye,2 reye,3 lear,4 rear,5 lsho,6 rsho,7 lelb,8 relb,9 lwri,10 rwri,
# 11 lhip,12 rhip,13 lkne,14 rkne,15 lank,16 rank
N_COCO = 17
N_HALPE = 26


def _bbox_from_kpts(kpts: np.ndarray, scores: np.ndarray, thr: float = 0.3) -> np.ndarray | None:
    valid = scores > thr
    if valid.sum() < 4:
        return None
    pts = kpts[valid]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-9))


def _coco_to_halpe26(kpts: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expand a COCO-17 pose to Halpe-26 (only 17,18,19 are consumed downstream)."""
    hk = np.zeros((N_HALPE, 2), dtype=np.float32)
    hs = np.zeros((N_HALPE,), dtype=np.float32)
    hk[:N_COCO] = kpts
    hs[:N_COCO] = scores
    # 18 Neck = mid-shoulders(5,6); 19 Hip = mid-hips(11,12); 17 Head = mid-ears(3,4)
    hk[18] = (kpts[5] + kpts[6]) * 0.5
    hs[18] = min(scores[5], scores[6])
    hk[19] = (kpts[11] + kpts[12]) * 0.5
    hs[19] = min(scores[11], scores[12])
    ear_s = min(scores[3], scores[4])
    if ear_s > 0.2:
        hk[17] = (kpts[3] + kpts[4]) * 0.5
        hs[17] = ear_s
    else:  # fall back to nose if ears weak
        hk[17] = kpts[0]
        hs[17] = scores[0]
    return hk, hs


def _densify(track_frames: dict[int, tuple[np.ndarray, np.ndarray]], n_frames: int
             ) -> tuple[np.ndarray, np.ndarray]:
    """Turn a sparse {frame: (kpts17, scores17)} into dense (n_frames,17,2)/(n_frames,17)."""
    kk = np.full((n_frames, N_COCO, 2), np.nan, dtype=np.float32)
    ss = np.zeros((n_frames, N_COCO), dtype=np.float32)
    for f, (k, s) in track_frames.items():
        if 0 <= f < n_frames:
            kk[f] = k
            ss[f] = s
    # linear interpolation of each coordinate across missing frames; hold at ends
    idx = np.arange(n_frames)
    for j in range(N_COCO):
        for c in range(2):
            col = kk[:, j, c]
            known = ~np.isnan(col)
            if known.sum() == 0:
                col[:] = 0.0
            elif known.sum() < n_frames:
                col[~known] = np.interp(idx[~known], idx[known], col[known])
            kk[:, j, c] = col
    return kk, ss


def generate_mb_input(
    rally_dir: Path | str,
    n_frames: int,
    cfg: config.PipelineConfig,
    force: bool = False,
) -> Path:
    """Detect + track players and write mb_input.json. Returns its path.

    Raises StageError if fewer than two persistent player tracks are found.
    """
    rally_dir = Path(rally_dir).resolve()
    rally_mp4 = rally_dir / "rally.mp4"
    out_json = rally_dir / "mb_input.json"
    if out_json.exists() and not force:
        LOG.info("[pose2d] mb_input.json exists, skipping (%s)", rally_dir.name)
        return out_json
    if not rally_mp4.exists():
        raise StageError(f"rally.mp4 missing in {rally_dir}")

    import cv2  # local imports so the module loads even before the ML stack is ready
    try:
        from rtmlib import Body
    except ImportError as exc:  # pragma: no cover
        raise StageError("rtmlib not installed; run scripts/setup_env.py") from exc

    device = "cuda"
    body = Body(mode=cfg.rtmpose_mode, backend="onnxruntime", device=device)

    cap = cv2.VideoCapture(str(rally_mp4))
    # tracks: id -> {"box": last_box, "frames": {frame: (kpts17, scores17)}, "last": frame}
    tracks: dict[int, dict] = {}
    next_id = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        keypoints, scores = body(frame)  # (N,17,2), (N,17)
        dets = []
        for k, s in zip(np.asarray(keypoints), np.asarray(scores)):
            box = _bbox_from_kpts(k, s)
            if box is not None and float(np.mean(s)) >= cfg.det_score_threshold * 0.5:
                dets.append((box, k.astype(np.float32), s.astype(np.float32)))
        # greedy IoU association to existing tracks
        assigned = set()
        for tid, tr in tracks.items():
            best, best_iou = -1, 0.3
            for di, (box, _k, _s) in enumerate(dets):
                if di in assigned:
                    continue
                iou = _iou(tr["box"], box)
                if iou > best_iou:
                    best, best_iou = di, iou
            if best >= 0:
                box, k, s = dets[best]
                tr["box"], tr["last"] = box, frame_idx
                tr["frames"][frame_idx] = (k, s)
                assigned.add(best)
        for di, (box, k, s) in enumerate(dets):
            if di in assigned:
                continue
            tracks[next_id] = {"box": box, "frames": {frame_idx: (k, s)}, "last": frame_idx}
            next_id += 1
        frame_idx += 1
    cap.release()
    n_frames = n_frames or frame_idx

    # rank tracks by coverage * median area; keep persistent ones
    def _score(tr: dict) -> float:
        cover = len(tr["frames"])
        if cover < cfg.min_player_track_frames:
            return -1.0
        areas = []
        for (k, s) in tr["frames"].values():
            b = _bbox_from_kpts(k, s)
            if b is not None:
                areas.append((b[2] - b[0]) * (b[3] - b[1]))
        med_area = float(np.median(areas)) if areas else 0.0
        return cover * med_area

    ranked = sorted(tracks.items(), key=lambda kv: _score(kv[1]), reverse=True)
    ranked = [(tid, tr) for tid, tr in ranked if _score(tr) > 0]
    if len(ranked) < 2:
        raise StageError(
            f"Only {len(ranked)} persistent player track(s) in {rally_dir.name}; "
            "need 2 (broadcast angle / crowd may be interfering)"
        )
    two = [ranked[0], ranked[1]]

    # assign idx by vertical position: far side (smaller mean y) -> 0, near -> 1
    def _mean_y(tr: dict) -> float:
        ys = [np.nanmean(k[:, 1]) for (k, s) in tr["frames"].values()]
        return float(np.mean(ys))
    two.sort(key=lambda kv: _mean_y(kv[1]))

    instances = []
    for player_idx, (tid, tr) in enumerate(two):
        kk, ss = _densify(tr["frames"], n_frames)
        for f in range(n_frames):
            hk, hs = _coco_to_halpe26(kk[f], ss[f])
            kps_flat = np.concatenate([hk, hs[:, None]], axis=1).reshape(-1).tolist()
            x1, y1 = hk[:N_COCO].min(axis=0)
            x2, y2 = hk[:N_COCO].max(axis=0)
            instances.append({
                "image_id": f"{f}.jpg",
                "category_id": 1,
                "keypoints": kps_flat,          # 26*(x,y,score)
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "idx": player_idx,
            })
    # MotionBERT reads items in file order and groups by idx; keep player 0 then 1,
    # each already in frame order.
    out_json.write_text(json.dumps(instances), encoding="utf-8")
    LOG.info("[pose2d] %s: 2 players, %d frames, %d instances",
             rally_dir.name, n_frames, len(instances))
    return out_json
