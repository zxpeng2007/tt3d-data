# Dataset schema

Each reconstructed rally is one directory under `data/dataset/rallies/<match_id>/<rally_id>/`.
The consolidated dataset is described by `data/dataset/manifest.parquet` (one row per rally) plus a
top-level `data/dataset/dataset_card.json`.

## Per-rally files

### `meta.json`
```json
{
  "match_id": "<youtube_video_id>",
  "rally_id": "r0007",
  "source_url": "https://www.youtube.com/watch?v=...",
  "start_frame": 12840, "end_frame": 13102,
  "fps": 25.0, "width": 1920, "height": 1080,
  "n_frames": 262,
  "camera_static": true,
  "quality": { "calib_ok": true, "n_players_tracked": 2, "ball_coverage": 0.83 }
}
```

### `camera.yaml`
Calibrated monocular camera for the rally (from TT3D calibration). Intrinsics (focal length, optical
center) and extrinsics (rotation + translation, table→camera). For static cameras a single filtered
pose; for moving cameras, `cam_cal.csv` holds the per-frame track.

### Ball
- **`ball_traj_2D.csv`** — per detected frame: `frame,u,v,score,blur_angle,blur_len` (image-space
  ball position in pixels + BlurBall motion-blur cue). Missing detections are gaps in `frame`.
- **`ball_traj_3D.csv`** — reconstructed world-frame trajectory: `t,x,y,z,vx,vy,vz,wx,wy,wz`
  (position m, velocity m/s, spin rad/s) sampled per frame, with `bounce` flags at contacts.

### Body pose
- **`p0_3d.npy`, `p1_3d.npy`** — shape `(n_frames, 17, 3)`, float32. 3D joints (COCO-17 / H36M-17
  order — see `pose_format` in `meta.json`) in the **table/world frame** (metres), after
  reprojection-error alignment. `NaN` rows where a player is untracked.
- (intermediate `player_0.npy`/`player_1.npy` are camera-frame MotionBERT outputs, pre-alignment.)

### Table
The table is the world-frame origin. Its geometry is the standard ITTF table
(2.74 m × 1.525 m, net 0.1525 m) placed at the origin; the calibrated `camera.yaml` defines where the
camera sits relative to it. `table.json` records the corner world coordinates used for calibration.

## Manifest columns (`manifest.parquet`)
`match_id, rally_id, source_url, n_frames, fps, calib_ok, n_players, ball_coverage,
has_ball_3d, has_pose_p0, has_pose_p1, path`
